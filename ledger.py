import streamlit as st
import pandas as pd
import io
import re
import pdfplumber
import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract

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
        images = convert_from_bytes(pdf_file_bytes) 
        
        full_text = []
        for img in images:
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
                for img_data in page.images:
                    if 'stream' in img_data:
                        all_extracted_images.append(img_data['stream'].get_data())
                
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
            
            if all_extracted_tables:
                final_df = pd.concat(all_extracted_tables, ignore_index=True)
                final_df.dropna(axis=1, how='all', inplace=True)
                return final_df, all_extracted_images
            
            st.warning("No structured data could be extracted from the PDF. Displaying as raw text lines.")
            pdf_file_buffer.seek(0)
            with pdfplumber.open(pdf_file_buffer) as pdf:
                text_content = ""
                for page in pdf.pages:
                    text_content += page.extract_text() + "\n"
            lines = [line.strip() for line in text_content.splitlines() if line.strip()]
            return pd.DataFrame(lines, columns=["Text_Content"]), all_extracted_images
    
    except Exception as e:
        st.error(f"An unexpected error occurred during native PDF parsing or OCR: {e}")
        pdf_file_buffer.seek(0)
        with pdfplumber.open(pdf_file_buffer) as pdf:
            text_content = ""
            for page in pdf.pages:
                text_content += page.extract_text() + "\n"
        lines = [line.strip() for line in text_content.splitlines() if line.strip()]
        return pd.DataFrame(lines, columns=["Text_Content"]), all_extracted_images

# --- New Function for Balance Sheet Generation ---

def create_balance_sheet(df):
    """
    Creates a simple balance sheet based on a cleaned DataFrame.
    Assumes the DataFrame has a column that can be classified into
    Assets, Liabilities, and Equity based on keywords.
    """
    # 1. Identify the 'amount' column (from the original cleaning logic)
    amount_col = None
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['amount', 'value', 'price', 'balance', 'cost', 'revenue']):
            amount_col = col
            break
            
    if amount_col is None:
        return pd.DataFrame({'Message': ['No valid amount column found to create a balance sheet.']})
        
    # 2. Identify a 'category' or 'type' column for classification
    category_col = None
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['category', 'type', 'account']):
            category_col = col
            break

    if category_col is None:
        return pd.DataFrame({'Message': ['No valid category column found to create a balance sheet.']})

    # 3. Define keywords for each financial category
    assets_keywords = ['cash', 'accounts receivable', 'inventory', 'property', 'equipment', 'investment', 'asset']
    liabilities_keywords = ['accounts payable', 'loan', 'debt', 'liability', 'mortgage', 'payable']
    equity_keywords = ['retained earnings', 'common stock', 'equity']
    
    # 4. Initialize totals
    total_assets = 0
    total_liabilities = 0
    total_equity = 0
    
    # 5. Classify and sum the values
    for index, row in df.iterrows():
        try:
            category = str(row[category_col]).lower()
            amount = row[amount_col]
            
            if pd.isna(amount):
                continue
            
            if any(keyword in category for keyword in assets_keywords):
                total_assets += amount
            elif any(keyword in category for keyword in liabilities_keywords):
                total_liabilities += amount
            elif any(keyword in category for keyword in equity_keywords):
                total_equity += amount
        except KeyError:
            continue
            
    # 6. Create the balance sheet DataFrame
    balance_sheet_df = pd.DataFrame({
        'Category': ['Total Assets', 'Total Liabilities', 'Total Equity'],
        'Amount': [total_assets, total_liabilities, total_equity]
    })
    
    # Add a final row to verify the balance sheet equation
    balance_check = total_assets - (total_liabilities + total_equity)
    balance_sheet_df.loc[len(balance_sheet_df)] = ['Balance Check (Assets - Liabilities - Equity)', balance_check]
    
    return balance_sheet_df

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
    type=["csv", "xlsx", "xls", "txt", "pdf", "png", "jpg", "jpeg"],
    help="Supported formats: CSV, Excel (.xlsx, .xls), Text (.txt), PDF, and Image files (.png, .jpg, .jpeg)."
)

# --- Main App Logic (runs only when a file is uploaded) ---
if uploaded_file is not None:
    file_name_display = uploaded_file.name
    file_extension = file_name_display.split('.')[-1].lower()

    st.info(f"Processing file: **{file_name_display}**")

    # Initialize dataframes and image lists
    df_original = pd.DataFrame()
    df_cleaned = pd.DataFrame()
    extracted_images_from_pdf = []
    uploaded_image_file_bytes = None

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
            
            if not df_original.empty:
                st.success("File loaded successfully!")
                st.subheader("Original Data Preview")
                st.dataframe(df_original.head())
                st.write(f"Shape: {df_original.shape[0]} rows, {df_original.shape[1]} columns")

        elif file_extension == "pdf":
            st.warning("Attempting advanced Python-native PDF table/image extraction and OCR. This may take a moment.")
            with st.spinner('Extracting data from PDF...'):
                df_original, extracted_images_from_pdf = pdf_to_dataframe_and_images(uploaded_file)
            
            if not df_original.empty:
                st.success("PDF data extracted and loaded successfully!")
                st.subheader("Original Data Preview (from PDF)")
                st.dataframe(df_original.head())
                st.write(f"Shape: {df_original.shape[0]} rows, {df_original.shape[1]} columns")
        
        elif file_extension in ["png", "jpg", "jpeg"]:
            uploaded_image_file_bytes = uploaded_file.getvalue()
            st.success("Image file uploaded successfully!")
            st.info("Click 'Show Uploaded Image' to view it.")

    except Exception as e:
        st.error(f"An error occurred during file processing: {e}")
        st.info("Please ensure your file is correctly formatted for the selected type.")

    st.markdown("---")
    
    # --- Data Cleaning and Analysis Section (for data files only) ---
    if not df_original.empty:
        st.subheader("Cleaning Data...")
        with st.spinner('Applying cleaning rules...'):
            df_cleaned = perform_data_cleaning(df_original)
        
        st.success("Data cleaning complete!")
        st.subheader("Cleaned Data Preview")
        st.dataframe(df_cleaned.head())
        st.write(f"Shape: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")

        csv_export = df_cleaned.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Cleaned Data as CSV",
            data=csv_export,
            file_name=f"cleaned_{file_name_display.split('.')[0]}.csv",
            mime="text/csv",
            help="Download the processed data in CSV format."
        )

        st.markdown("---")
        
        balance_sheet_df = create_balance_sheet(df_cleaned)
        if not balance_sheet_df.empty and 'Message' not in balance_sheet_df.columns:
            with st.expander("📊 Generate Balance Sheet"):
                st.info("""
                This balance sheet is an **estimation** based on a keyword search.
                For accurate results, ensure your dataset has a column that can be
                identified as a financial 'Category' or 'Account' (e.g., 'cash', 'loan', 'stock'),
                and another for 'Amount' or 'Value'.
                """)
                st.dataframe(balance_sheet_df)
                st.markdown(
                    "The fundamental balance sheet equation is: **Assets = Liabilities + Equity**"
                )
                st.markdown(
                    "The final `Balance Check` should be as close to `0` as possible."
                )
        else:
             st.warning("Could not generate a balance sheet. " + balance_sheet_df.iloc[0]['Message'])

    # --- Image Display Section ---
    st.markdown("---")
    image_col1, image_col2 = st.columns(2)

    with image_col1:
        if extracted_images_from_pdf:
            if st.button("🖼️ Show Extracted Images (from PDF)"):
                for i, img_bytes in enumerate(extracted_images_from_pdf):
                    try:
                        st.image(img_bytes, caption=f"Image {i+1} from PDF", use_column_width=True)
                    except Exception as img_e:
                        st.warning(f"Could not display image {i+1}: {img_e}")
        
    with image_col2:
        if uploaded_image_file_bytes:
            if st.button("📸 Show Uploaded Image"):
                try:
                    st.image(uploaded_image_file_bytes, caption="Directly Uploaded Image", use_column_width=True)
                except Exception as img_e:
                    st.warning(f"Could not display uploaded image: {img_e}")
    
else:
    # This message is displayed when no file has been uploaded yet
    st.info("Please upload a file to get started.")

st.markdown("---")
st.markdown("""
<p style="font-size: 0.8em; color: grey;">
**Note on PDF Files:** This app provides the most robust Python-native solution for PDFs, combining text-based table extraction, OCR for image-based documents, and direct extraction of embedded images. However, perfect extraction from highly complex or poor-quality PDFs remains a challenging task.
</p>
""", unsafe_allow_html=True)