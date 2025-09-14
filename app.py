import streamlit as st
import pandas as pd
import io
import re
import pdfplumber
import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract

# --- Database Imports ---
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.types import String, Numeric, Date, Boolean, DateTime
import json # Used for serializing non-standard types if needed

# --- Helper Functions for Data Cleaning ---

def standardize_date(date_series):
    """
    Attempts to standardize various date formats to 'MM/DD/YYYY'.
    Invalid dates (or NaT) will be returned as empty strings after conversion.
    """
    return pd.to_datetime(date_series, errors='coerce').dt.strftime('%m/%d/%Y').fillna('')

def standardize_currency(currency_series):
    """
    Cleans and converts currency strings to numeric (float) values.
    Handles various currency symbols, commas, parenthetical negatives, and invalid values.
    Invalid values are replaced with np.nan.
    """
    cleaned_series = currency_series.astype(str).apply(lambda x: x.strip())
    
    cleaned_series = cleaned_series.apply(lambda s: '-' + s[1:-1] if s.startswith('(') and s.endswith(')') else s)
    
    cleaned_series = cleaned_series.str.replace(r'[$,€£¥]', '', regex=True)
    cleaned_series = cleaned_series.str.replace(r'\s+', '', regex=True)
    cleaned_series = cleaned_series.str.replace(',', '', regex=False)
    
    return pd.to_numeric(cleaned_series, errors='coerce')


def perform_data_cleaning(df):
    """
    Performs core financial data cleaning operations on a Pandas DataFrame.
    - Robustly handles missing, null, and NaN values.
    - Standardizes date and currency formats.
    - Trims whitespace from all string columns.
    - Converts 'Yes'/'No' to boolean.
    """
    cleaned_df = df.copy()

    for col in cleaned_df.columns:
        # Step 1: Normalize common missing value indicators to np.nan
        cleaned_df[col] = cleaned_df[col].replace({'': np.nan, ' ': np.nan, 'NA': np.nan, 'N/A': np.nan, '-': np.nan, 'NULL': np.nan, 'None': np.nan})
        
        col_lower = col.lower()

        # Step 2: Date Standardization
        if any(keyword in col_lower for keyword in ['date', 'day', 'month', 'year']):
            cleaned_df[col] = standardize_date(cleaned_df[col])
            
        # Step 3: Currency/Amount Standardization
        elif any(keyword in col_lower for keyword in ['amount', 'value', 'price', 'balance', 'cost', 'revenue']):
            cleaned_df[col] = standardize_currency(cleaned_df[col])
            cleaned_df[col] = cleaned_df[col].fillna(0)
        
        # Step 4: General Inconsistent Value Handling (after specific formats)
        else:
            if pd.api.types.is_object_dtype(cleaned_df[col]):
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
                if cleaned_df[col].str.lower().isin(['yes', 'no']).any():
                    cleaned_df[col] = cleaned_df[col].str.lower().map({'yes': True, 'no': False, 'nan': np.nan, '': np.nan})
                    cleaned_df[col] = cleaned_df[col].fillna(False)
                else:
                    temp_numeric = pd.to_numeric(cleaned_df[col], errors='coerce')
                    if temp_numeric.notna().all() and not temp_numeric.empty:
                        cleaned_df[col] = temp_numeric
                        cleaned_df[col] = cleaned_df[col].fillna(0)
                    else:
                        cleaned_df[col] = cleaned_df[col].fillna('')
            
            elif pd.api.types.is_numeric_dtype(cleaned_df[col]):
                cleaned_df[col] = cleaned_df[col].fillna(0)

    return cleaned_df

# --- OCR for Image-Based PDFs ---
def ocr_pdf_to_text(pdf_file_bytes):
    """
    Converts PDF pages to images and performs OCR to extract text.
    Returns concatenated text from all pages.
    """
    try:
        # Convert PDF bytes to a list of PIL images
        # IMPORTANT: Adjust poppler_path if it's different on your system
        # For deployment, consider setting POPPLER_PATH as an environment variable
        poppler_path_config = r'C:\Users\ranes\Downloads\poppler-24.02.0\Library\bin' # Default path, adjust if needed
        images = convert_from_bytes(pdf_file_bytes, poppler_path=poppler_path_config) 
        
        full_text = []
        for img in images:
            # Perform OCR on each image
            text = pytesseract.image_to_string(img)
            if text:
                full_text.append(text)
        
        return "\n".join(full_text)
    except Exception as e:
        st.error(f"OCR failed. Please ensure Tesseract and Poppler are installed and in PATH. Error: {e}")
        return ""

# --- PDF Text/Image Parsing Function (Python-Native) ---

def pdf_to_dataframe_and_images(pdf_file_buffer):
    """
    Attempts to extract structured data and embedded images from PDF.
    First tries pdfplumber's table extraction, then heuristic text parsing.
    If fails and suspects image-based PDF, attempts OCR.
    Returns: (DataFrame, list of image bytes)
    """
    pdf_bytes = pdf_file_buffer.getvalue() # Get bytes for OCR
    pdf_file_buffer.seek(0) # Reset buffer for pdfplumber

    all_extracted_tables = []
    all_extracted_images = []
    has_text_content = False

    try:
        with pdfplumber.open(pdf_file_buffer) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract embedded images
                for img_data in page.images:
                    if 'stream' in img_data:
                        all_extracted_images.append(img_data['stream'].get_data())

                # Check if page has extractable text
                if page.extract_text().strip():
                    has_text_content = True

                tables = page.extract_tables(table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "snap_tolerance": 3,
                    "intersection_tolerance": 3,
                    "text_tolerance": 3
                })

                if tables:
                    for table in tables:
                        if table and len(table) > 0:
                            headers = [h if h is not None else f"Column_{i}" for i, h in enumerate(table[0])]
                            data = table[1:]
                            df_table = pd.DataFrame(data, columns=headers)
                            df_table.dropna(axis=0, how='all', inplace=True)
                            if not df_table.empty:
                                all_extracted_tables.append(df_table)
            
            # If pdfplumber found no tables but there was text, try heuristic text parsing
            if not all_extracted_tables and has_text_content:
                st.warning("No tables found using pdfplumber's built-in logic. Attempting text-based heuristics.")
                pdf_file_buffer.seek(0)
                raw_full_text = ""
                with pdfplumber.open(pdf_file_buffer) as p:
                    for pg in p.pages:
                        raw_full_text += pg.extract_text(x_tolerance=2).strip() + "\n"
                
                if raw_full_text.strip():
                    lines = [line.strip() for line in raw_full_text.splitlines() if line.strip()]
                    for sep in [',', '\t', r'\s{2,}']:
                        try:
                            df_heuristic = pd.read_csv(io.StringIO("\n".join(lines)), sep=sep, engine='python')
                            if not df_heuristic.empty and df_heuristic.shape[1] > 1:
                                st.info(f"PDF text successfully parsed as structured data using '{sep}' heuristic.")
                                return df_heuristic, all_extracted_images
                        except Exception:
                            continue
        
        # Fallback to OCR if no structured text or tables found via pdfplumber
        if not all_extracted_tables:
            st.warning("No structured data found via text extraction. Attempting OCR for image-based PDFs. This can be slow.")
            ocr_text = ocr_pdf_to_text(pdf_bytes)
            if ocr_text.strip():
                st.info("Text successfully extracted via OCR. Attempting heuristic parsing.")
                lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
                for sep in [',', '\t', r'\s{2,}']:
                    try:
                        df_ocr = pd.read_csv(io.StringIO("\n".join(lines)), sep=sep, engine='python')
                        if not df_ocr.empty and df_ocr.shape[1] > 1:
                            st.info(f"OCR text successfully parsed as structured data using '{sep}' heuristic.")
                            return df_ocr, all_extracted_images
                    except Exception:
                        continue
                
                st.warning("OCR text could not be parsed into structured data. Displaying as plain OCR text lines.")
                return pd.DataFrame(lines, columns=["OCR_Text_Content"]), all_extracted_images
            else:
                st.error("OCR yielded no text. The PDF might be entirely blank or highly unreadable.")
                return pd.DataFrame(), all_extracted_images

    except Exception as e:
        st.error(f"An unexpected error occurred during native PDF parsing or OCR: {e}")
        # Final fallback to raw text extraction if everything above completely fails
        pdf_file_buffer.seek(0)
        with pdfplumber.open(pdf_file_buffer) as pdf:
            text_content = ""
            for page in pdf.pages:
                text_content += page.extract_text() + "\n"
        lines = [line.strip() for line in text_content.splitlines() if line.strip()]
        return pd.DataFrame(lines, columns=["Text_Content"]), all_extracted_images

    if all_extracted_tables:
        final_df = pd.concat(all_extracted_tables, ignore_index=True)
        final_df.dropna(axis=1, how='all', inplace=True)
        return final_df, all_extracted_images
    else:
        # If no tables or structured text could be extracted at all
        st.warning("No structured data could be extracted from the PDF. Displaying as raw text lines.")
        pdf_file_buffer.seek(0)
        with pdfplumber.open(pdf_file_buffer) as pdf:
            text_content = ""
            for page in pdf.pages:
                text_content += page.extract_text() + "\n"
        lines = [line.strip() for line in text_content.splitlines() if line.strip()]
        return pd.DataFrame(lines, columns=["Text_Content"]), all_extracted_images

# --- Database Integration Functions ---

# Use Streamlit secrets or environment variables for sensitive info in a real app
DB_CONNECTION_STRING = "postgresql+psycopg2://postgres:manisha20@localhost:5432/financial_data_db"

@st.cache_resource # Cache the database engine to prevent re-creation on every rerun
def get_db_engine():
    try:
        engine = create_engine(DB_CONNECTION_STRING)
        return engine
    except Exception as e:
        st.error(f"Could not connect to the database: {e}")
        return None

def store_dataframe_to_db(df, table_name):
    """
    Stores a DataFrame to a PostgreSQL table, creating/altering columns dynamically.
    """
    engine = get_db_engine()
    if engine is None:
        return False

    try:
        # Check if table exists
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()

        # Sanitize column names for SQL compatibility
        df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in df.columns]

        # Map Pandas dtypes to SQLAlchemy types
        dtype_mapping = {}
        for col in df.columns:
            if pd.api.types.is_float_dtype(df[col]):
                dtype_mapping[col] = Numeric
            elif pd.api.types.is_integer_dtype(df[col]):
                dtype_mapping[col] = Numeric # Use Numeric for integers too, or Integer if precise
            elif pd.api.types.is_bool_dtype(df[col]):
                dtype_mapping[col] = Boolean
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                dtype_mapping[col] = DateTime # Or Date if you only need date
            else: # Default to String for object/string types
                dtype_mapping[col] = String(255) # Set a reasonable default string length

        if table_name in existing_tables:
            # If table exists, check for new columns and add them
            st.info(f"Table '{table_name}' already exists. Checking for new columns...")
            with engine.connect() as connection:
                existing_columns = [col['name'] for col in inspector.get_columns(table_name)]
                for col_name, col_type in dtype_mapping.items():
                    if col_name not in existing_columns:
                        try:
                            # Add new column
                            add_column_sql = text(f'ALTER TABLE "{table_name}" ADD COLUMN "{col_name}" {col_type.compile(dialect=engine.dialect)}')
                            connection.execute(add_column_sql)
                            connection.commit()
                            st.success(f"Added new column: {col_name} ({col_type}) to '{table_name}'")
                        except Exception as e:
                            st.warning(f"Could not add column {col_name}: {e}")
                
                # Check for columns in DB but not in current DF (and warn or handle)
                for db_col in existing_columns:
                    if db_col not in df.columns:
                        st.warning(f"Column '{db_col}' exists in database table '{table_name}' but not in current data. It will be left as is.")

            # Append new data
            df.to_sql(table_name, engine, if_exists='append', index=False, dtype=dtype_mapping)
            st.success(f"Data appended to existing table '{table_name}' successfully!")
        else:
            # If table does not exist, create it and insert data
            df.to_sql(table_name, engine, if_exists='replace', index=False, dtype=dtype_mapping)
            st.success(f"New table '{table_name}' created and data stored successfully!")
        
        return True
    except Exception as e:
        st.error(f"Error storing data to database: {e}")
        return False

# --- Streamlit UI Setup ---

st.set_page_config(layout="wide", page_title="Financial Data Cleaner", page_icon="💰")

st.title("💰 Financial Data Cleaner")
st.markdown("""
Upload your financial datasets (CSV, Excel, Text, PDF) or image files for processing.
The app will clean financial data, and provide options to view extracted images from PDFs or directly uploaded images.
""")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload your financial dataset or an image file",
    type=["csv", "xlsx", "xls", "txt", "pdf", "png", "jpg", "jpeg"], # Added image types
    help="Supported formats: CSV, Excel (.xlsx, .xls), Text (.txt), PDF (Python-native heuristic table detection + OCR + image extraction), and Image files (.png, .jpg, .jpeg)."
)

df_original = pd.DataFrame()
df_cleaned = pd.DataFrame()
# Initialize extracted_images_from_pdf in session state
if 'extracted_images_from_pdf' not in st.session_state:
    st.session_state.extracted_images_from_pdf = []
# Initialize directly uploaded image in session state
if 'uploaded_image_file' not in st.session_state:
    st.session_state.uploaded_image_file = None
# Initialize a flag to control image display
if 'show_images_clicked' not in st.session_state:
    st.session_state.show_images_clicked = False

file_name_display = "No file uploaded"

# Input for table name
st.sidebar.header("Database Options")
database_table_name = st.sidebar.text_input("Enter Database Table Name:", value="financial_data")
if not database_table_name:
    st.sidebar.warning("Please enter a table name to store the data.")

if uploaded_file is not None:
    file_name_display = uploaded_file.name
    file_extension = file_name_display.split('.')[-1].lower()

    st.info(f"Processing file: **{file_name_display}**")

    # Reset image display flag and images when a new file is uploaded
    st.session_state.show_images_clicked = False
    st.session_state.extracted_images_from_pdf = []
    st.session_state.uploaded_image_file = None # Clear previously uploaded image

    try:
        if file_extension in ["csv", "xlsx", "xls", "txt"]:
            if file_extension == "csv":
                df_original = pd.read_csv(uploaded_file)
            elif file_extension in ["xlsx", "xls"]:
                df_original = pd.read_excel(uploaded_file)
            elif file_extension == "txt":
                try:
                    df_original = pd.read_csv(uploaded_file, sep=None, engine='python')
                    if df_original.empty or df_original.shape[1] <= 1:
                         raise ValueError("Not structured TXT")
                    st.info("TXT file successfully parsed as structured data.")
                except Exception:
                    st.warning("Could not parse .txt as structured data. Reading as plain text lines.")
                    text_content = uploaded_file.read().decode('utf-8')
                    lines = [line.strip() for line in text_content.splitlines() if line.strip()]
                    df_original = pd.DataFrame(lines, columns=["Text_Content"])
            
            # For data files, perform cleaning and display
            if not df_original.empty:
                st.success("File loaded successfully!")
                st.subheader("Original Data Preview")
                st.dataframe(df_original.head())
                st.write(f"Shape: {df_original.shape[0]} rows, {df_original.shape[1]} columns")

                st.markdown("---")
                st.subheader("Cleaning Data...")
                with st.spinner('Applying cleaning rules...'):
                    df_cleaned = perform_data_cleaning(df_original)
                
                st.success("Data cleaning complete!")
                st.subheader("Cleaned Data Preview")
                st.dataframe(df_cleaned.head())
                st.write(f"Shape: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")

                # Database Storage Button
                if not df_cleaned.empty and database_table_name:
                    if st.button(f"💾 Store Cleaned Data to '{database_table_name}'"):
                        with st.spinner(f"Storing data to database table '{database_table_name}'..."):
                            success = store_dataframe_to_db(df_cleaned, database_table_name)
                            if success:
                                st.success("Data successfully stored in the database!")
                            else:
                                st.error("Failed to store data in the database.")
                elif not database_table_name:
                    st.warning("Please provide a database table name in the sidebar to store data.")


                csv_export = df_cleaned.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Cleaned Data as CSV",
                    data=csv_export,
                    file_name=f"cleaned_{file_name_display.split('.')[0]}.csv",
                    mime="text/csv",
                    help="Download the processed data in CSV format."
                )
            else:
                st.warning("The uploaded file is empty or could not be parsed into a DataFrame.")

        elif file_extension == "pdf":
            st.warning("Attempting advanced Python-native PDF table/image extraction and OCR. This may take a moment.")
            df_original, current_extracted_images = pdf_to_dataframe_and_images(uploaded_file)
            st.session_state.extracted_images_from_pdf = current_extracted_images # Store in session state
            
            if not df_original.empty:
                st.success("PDF data extracted and loaded successfully!")
                st.subheader("Original Data Preview (from PDF)")
                st.dataframe(df_original.head())
                st.write(f"Shape: {df_original.shape[0]} rows, {df_original.shape[1]} columns")

                st.markdown("---")
                st.subheader("Cleaning Data...")
                with st.spinner('Applying cleaning rules...'):
                    df_cleaned = perform_data_cleaning(df_original)
                
                st.success("Data cleaning complete!")
                st.subheader("Cleaned Data Preview")
                st.dataframe(df_cleaned.head())
                st.write(f"Shape: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")

                # Database Storage Button for PDF data
                if not df_cleaned.empty and database_table_name:
                    if st.button(f"💾 Store Cleaned Data to '{database_table_name}'"):
                        with st.spinner(f"Storing data to database table '{database_table_name}'..."):
                            success = store_dataframe_to_db(df_cleaned, database_table_name)
                            if success:
                                st.success("Data successfully stored in the database!")
                            else:
                                st.error("Failed to store data in the database.")
                elif not database_table_name:
                    st.warning("Please provide a database table name in the sidebar to store data.")


                csv_export = df_cleaned.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Cleaned Data as CSV",
                    data=csv_export,
                    file_name=f"cleaned_{file_name_display.split('.')[0]}.csv",
                    mime="text/csv",
                    help="Download the processed data in CSV format."
                )
            else:
                st.warning("No structured data could be extracted from the PDF for cleaning.")

        elif file_extension in ["png", "jpg", "jpeg"]: # Handle direct image uploads
            st.session_state.uploaded_image_file = uploaded_file.getvalue() # Store image bytes
            st.success("Image file uploaded successfully!")
            st.info("Click 'Show Uploaded Image' to view it.")
            # No DataFrame cleaning for direct image uploads
            df_original = pd.DataFrame() # Ensure no data table is displayed for image files
            df_cleaned = pd.DataFrame() # Ensure no data table is displayed for image files
        
        else:
            st.warning("Unsupported file type. Please upload a supported document or image.")

    except Exception as e:
        st.error(f"An error occurred during file processing: {e}")
        st.info("Please ensure your file is correctly formatted for the selected type.")

st.markdown("---")
# Image Display Options
image_col1, image_col2 = st.columns(2)

with image_col1:
    # Button for images extracted from PDF
    if st.session_state.extracted_images_from_pdf:
        if st.button("🖼️ Show Extracted Images (from PDF)"):
            st.session_state.show_images_clicked = True
    elif uploaded_file is not None and uploaded_file.name.lower().endswith('.pdf'):
        st.info("No embedded images were found in the uploaded PDF.")

with image_col2:
    # Button for directly uploaded image files
    if st.session_state.uploaded_image_file:
        if st.button("📸 Show Uploaded Image"):
            st.session_state.show_images_clicked = True
    elif uploaded_file is not None and uploaded_file.name.split('.')[-1].lower() in ["png", "jpg", "jpeg"]:
        st.info("Uploaded image ready to be shown.")

# Display images based on the flag
if st.session_state.show_images_clicked:
    if st.session_state.extracted_images_from_pdf:
        st.subheader("Extracted Images from PDF")
        for i, img_bytes in enumerate(st.session_state.extracted_images_from_pdf):
            try:
                st.image(img_bytes, caption=f"Image {i+1} from PDF", use_column_width=True)
            except Exception as img_e:
                st.warning(f"Could not display image {i+1}: {img_e}")
    
    if st.session_state.uploaded_image_file:
        st.subheader("Uploaded Image File")
        try:
            st.image(st.session_state.uploaded_image_file, caption="Directly Uploaded Image", use_column_width=True)
        except Exception as img_e:
            st.warning(f"Could not display uploaded image: {img_e}")
    
    if not st.session_state.extracted_images_from_pdf and not st.session_state.uploaded_image_file:
        st.info("No images to display.")


st.markdown("---")
st.markdown("""
<p style="font-size: 0.8em; color: grey;">
**Note on PDF Files:** This app provides the most robust Python-native solution for PDFs, combining text-based table extraction, OCR for image-based documents, and direct extraction of embedded images. However, perfect extraction from highly complex or poor-quality PDFs remains a challenging task.
</p>
""", unsafe_allow_html=True)

