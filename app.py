import streamlit as st
import os
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud
import pandas as pd

# Import our custom modules
from src.inference import predict_document
from src.extraction import extract_information
from src.summarization import generate_summary
from src.utils import (
    init_db, 
    save_to_db, 
    get_db_history, 
    calculate_text_metrics, 
    delete_db_entries, 
    convert_pdf_to_image
)

# Initialize the database immediately
init_db()

# 1. Page Config
st.set_page_config(page_title="DocuMind AI", page_icon="📄", layout="wide")

# 2. Custom CSS for "Neon/Dark" Look
def load_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #111827;
            border-right: 1px solid #374151;
        }

        /* Card Styling */
        div[data-testid="stMetric"] {
            background-color: #1F2937;
            border: 1px solid #374151;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        div[data-testid="stMetric"] label {
            color: #9CA3AF !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #F3F4F6 !important;
        }

        /* Button Styling */
        div.stButton > button:first-child {
            background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        div.stButton > button:first-child:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(124, 58, 237, 0.3);
        }

        /* File Uploader Styling */
        div[data-testid="stFileUploader"] {
            border: 1px dashed #4B5563;
            border-radius: 10px;
            padding: 20px;
            background-color: #1F2937;
        }
        div[data-testid="stFileUploader"]:hover {
            border-color: #8B5CF6;
        }

        h1, h2, h3 {
            color: #F3F4F6 !important;
        }
        
        div[data-testid="stTable"] {
            color: #E5E7EB;
        }
        </style>
    """, unsafe_allow_html=True)

load_css()

# 3. Sidebar Navigation
with st.sidebar:
    st.title("🧩 DocuMind AI")
    st.markdown("---")
    page = st.radio("Navigate to:", ["Analysis Dashboard", "History Log", "System Analytics"])
    st.markdown("---")
    st.caption("v1.0 | Powered by LayoutLM & SpaCy")

# ==========================================
# PAGE 1: ANALYSIS DASHBOARD
# ==========================================
if page == "Analysis Dashboard":
    st.markdown("""
    <h1 style='text-align: center; background: -webkit-linear-gradient(45deg, #60A5FA, #A78BFA); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
    Intelligent Document Analysis
    </h1>
    """, unsafe_allow_html=True)
    
    st.write("") # Spacer

    # --- UPLOAD SECTION ---
    st.container()
    uploaded_file = st.file_uploader("📂 Upload Document (PDF, JPG, PNG)", type=["jpg", "png", "jpeg", "tif", "pdf"])

    if uploaded_file is not None:
        st.markdown("---")
        
        # 1. SAVE THE FILE (Required for conversion)
        if not os.path.exists("data"):
            os.makedirs("data")
            
        file_ext = uploaded_file.name.split(".")[-1].lower()
        temp_path = os.path.join("data", f"temp_upload.{file_ext}")
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # 2. HANDLE PDF vs IMAGE
        display_image_path = temp_path 
        
        if file_ext == "pdf":
            with st.spinner("Converting PDF to Image for AI..."):
                converted_path = convert_pdf_to_image(temp_path)
                if converted_path:
                    display_image_path = converted_path
                else:
                    st.error("Failed to convert PDF.")
                    st.stop()

        # Layout: Image on Left, Actions on Right
        col1, col2 = st.columns([1, 2])
        
        with col1:
            image = Image.open(display_image_path)
            st.image(image, caption='Document Preview', width="stretch")
            
        with col2:
            st.subheader("Processing Options")
            st.info("Document detected. Ready to analyze.")
            
            # Primary Action Button
            analyze_btn = st.button('🚀 Analyze & Archive Document', use_container_width=True, type="primary")

        # Logic
        if analyze_btn:
            with st.spinner('🔍 Scanning & Processing...'):
                
                # Predict using the IMAGE path
                label, confidence, extracted_text = predict_document(display_image_path)
                
                # --- LABEL FIX (Optional) ---
                label_map = {"LABEL_0": "Resume", "LABEL_1": "Email"}
                if label in label_map:
                   label = label_map[label]
                # ----------------------------
                
                # Generate Summary
                summary = generate_summary(extracted_text)

                # Save to Database
                db_msg = save_to_db(uploaded_file, label, confidence, extracted_text, summary)
                st.toast(db_msg, icon="🗄️")
                
                # Save to Session State
                st.session_state['analyzed'] = True
                st.session_state['label'] = label
                st.session_state['confidence'] = confidence
                st.session_state['text'] = extracted_text
                st.session_state['summary'] = summary 

        # --- RESULTS SECTION ---
        if st.session_state.get('analyzed'):
            st.markdown("---")
            
            # 1. Classification Banner
            st.subheader("1. Classification Result")
            st.markdown(f"""
            <div style="background-color: #3730A3; padding: 15px; border-radius: 10px; border-left: 5px solid #818CF8;">
                <h3 style="margin:0; color: white;">Category: {st.session_state['label'].upper()}</h3>
                <p style="margin:0; color: #C7D2FE;">Confidence Score: {st.session_state['confidence']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("") 

            # 2. Text Metrics
            st.subheader("📝 Text Metrics")
            metrics = calculate_text_metrics(st.session_state['text'])
            if metrics:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Word Count", metrics["Word Count"])
                c2.metric("Sentences", metrics["Sentence Count"])
                c3.metric("Avg Word Len", metrics["Avg Word Length"])
                c4.metric("Readability", metrics["Readability Score (ARI)"])
            
            st.write("") 
            
            # Define Columns Correctly BEFORE using them
            col_left, col_right = st.columns(2)
            
            # 3. Extraction (Left)
            with col_left:
                st.subheader("2. Extracted Entities")
                details = extract_information(st.session_state['text'], st.session_state['label'])
                if details:
                    st.table(pd.DataFrame(list(details.items()), columns=["Field", "Value"]))
                else:
                    st.warning("No specific patterns found.")

            # 4. Summarization (Right)
            with col_right:
                st.subheader("3. AI Summary")
                if 'summary' in st.session_state:
                    st.success(st.session_state['summary'])
                else:
                    st.info("Summary available after analysis.")

            # 5. Visual Analytics
            st.subheader("4. Context Cloud")
            try:
                wordcloud = WordCloud(width=1000, height=400, background_color='#1F2937', colormap='cool').generate(st.session_state['text'])
                fig, ax = plt.subplots(figsize=(10, 4))
                fig.patch.set_alpha(0)
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            except:
                st.write("Not enough text for visualization.")

# ==========================================
# PAGE 2: HISTORY LOG
# ==========================================
elif page == "History Log":
    st.title("📜 Database History")
    st.write("Manage your archived document logs below.")
    
    df_history = get_db_history()
    
    if not df_history.empty:
        df_history.insert(0, "Select", False)
        
        edited_df = st.data_editor(
            df_history,
            column_config={
                "Select": st.column_config.CheckboxColumn("Delete?", width="small"),
                "upload_date": st.column_config.DatetimeColumn("Date", format="D MMM YYYY, h:mm a"),
                "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1.0, format="%.2f"),
                "filename": st.column_config.TextColumn("Filename"),
                "category": st.column_config.TextColumn("Category"),
                "summary": st.column_config.TextColumn("Summary", width="medium"),
            },
            disabled=["upload_date", "filename", "category", "confidence", "summary"],
            use_container_width=True,
            hide_index=True,
            height=600
        )
        
        rows_to_delete = edited_df[edited_df.Select].index
        
        if len(rows_to_delete) > 0:
            st.warning(f"⚠️ You have selected {len(rows_to_delete)} document(s) to delete.")
            if st.button("🗑️ Delete Selected", type="primary"):
                # Get the actual Database IDs
                ids_to_delete = edited_df.loc[rows_to_delete, "id"].tolist()
                success = delete_db_entries(ids_to_delete)
                if success:
                    st.success("Entries deleted successfully!")
                    st.rerun()
                else:
                    st.error("Error deleting entries.")
    else:
        st.info("No documents processed yet.")

# ==========================================
# PAGE 3: ANALYTICS
# ==========================================
elif page == "System Analytics":
    st.title("📊 System Analytics")
    
    df_history = get_db_history()
    if not df_history.empty:
        colA, colB = st.columns(2)
        
        with colA:
            st.subheader("Uploads by Category")
            st.bar_chart(df_history['category'].value_counts())
            
        with colB:
            st.subheader("AI Confidence Trend")
            try:
                st.line_chart(df_history['confidence'])
            except:
                st.write("Insufficient data for trend analysis.")
    else:
        st.info("No data available. Process some documents first!")




# # Add this to your imports
# from src.utils import init_db, save_to_db, get_db_history
# from src.summarization import generate_summary  # Ensure this is imported

# # Initialize the database immediately
# init_db()

# import streamlit as st
# import os
# import matplotlib.pyplot as plt
# from PIL import Image
# from wordcloud import WordCloud
# import pandas as pd

# # Import our custom modules
# from src.inference import predict_document
# from src.extraction import extract_information
# from src.summarization import generate_summary
# # from src.utils import save_and_log, get_history, calculate_text_metrics, delete_history_entries
# from src.utils import init_db, save_to_db, get_db_history, calculate_text_metrics, delete_db_entries

# # 1. Page Config
# st.set_page_config(page_title="DocuMind AI", page_icon="📄", layout="wide")

# # 2. Custom CSS for "Neon/Dark" Look
# def load_css():
#     st.markdown("""
#         <style>
#         /* Global Font and Background adjustments */
#         @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
#         html, body, [class*="css"] {
#             font-family: 'Inter', sans-serif;
#         }

#         /* Sidebar Styling */
#         section[data-testid="stSidebar"] {
#             background-color: #111827; /* Darker sidebar */
#             border-right: 1px solid #374151;
#         }

#         /* Card Styling for Metrics */
#         div[data-testid="stMetric"] {
#             background-color: #1F2937; /* Dark card background */
#             border: 1px solid #374151;
#             padding: 15px;
#             border-radius: 10px;
#             box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
#         }
#         div[data-testid="stMetric"] label {
#             color: #9CA3AF !important; /* Muted text for labels */
#         }
#         div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
#             color: #F3F4F6 !important; /* Bright text for values */
#         }

#         /* Custom Button Styling - "Neon" primary button */
#         div.stButton > button:first-child {
#             background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
#             color: white;
#             border: none;
#             padding: 0.5rem 1rem;
#             border-radius: 8px;
#             font-weight: 600;
#             transition: all 0.3s ease;
#         }
#         div.stButton > button:first-child:hover {
#             transform: translateY(-2px);
#             box-shadow: 0 10px 15px -3px rgba(124, 58, 237, 0.3);
#         }

#         /* File Uploader Styling */
#         div[data-testid="stFileUploader"] {
#             border: 1px dashed #4B5563;
#             border-radius: 10px;
#             padding: 20px;
#             background-color: #1F2937;
#         }
#         div[data-testid="stFileUploader"]:hover {
#             border-color: #8B5CF6; /* Neon purple on hover */
#         }

#         /* Headers and Titles */
#         h1, h2, h3 {
#             color: #F3F4F6 !important;
#         }
        
#         /* Table Styling */
#         div[data-testid="stTable"] {
#             color: #E5E7EB;
#         }
#         </style>
#     """, unsafe_allow_html=True)

# load_css()

# # 3. Sidebar Navigation
# with st.sidebar:
#     # You can add your Logo here if you have one
#     # st.image("logo.png", width=200) 
#     st.title("🧩 DocuMind AI")
#     st.markdown("---")
#     page = st.radio("Navigate to:", ["Analysis Dashboard", "History Log", "System Analytics"])
#     st.markdown("---")
#     st.caption("v1.0 | Powered by LayoutLM & SpaCy")

# # ==========================================
# # PAGE 1: ANALYSIS DASHBOARD
# # ==========================================
# if page == "Analysis Dashboard":
#     # Header Section with a gradient text effect (optional HTML hack)
#     st.markdown("""
#     <h1 style='text-align: center; background: -webkit-linear-gradient(45deg, #60A5FA, #A78BFA); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
#     Intelligent Document Analysis
#     </h1>
#     """, unsafe_allow_html=True)
    
#     st.write("") # Spacer

#     # --- UPLOAD SECTION ---
#     st.container()
#     # uploaded_file = st.file_uploader("📂 Upload Document (JPG, PNG, TIF)", type=["jpg", "png", "tif", "jpeg"])
#     uploaded_file = st.file_uploader("📂 Upload Document (PDF, JPG, PNG)", type=["jpg", "png", "jpeg", "pdf"])

#     if uploaded_file is not None:
#         st.markdown("---")
#         # Layout: Image on Left, Actions on Right
#         col1, col2 = st.columns([1, 2])
        
#         with col1:
#             image = Image.open(uploaded_file)
#             st.image(image, caption='Document Preview', use_container_width=True)
#             # NEW CODE (Fixes the warning)
            
#         with col2:
#             st.subheader("Processing Options")
#             st.info("Document detected. The system will now extract text, classify the document type, and generate a summary.")
            
#             # Primary Action Button
#             analyze_btn = st.button('🚀 Analyze & Archive Document', use_container_width=True, type="primary")

#         # Logic
#         # Logic
#         if analyze_btn:
#             with st.spinner('🔍 Scanning & Processing...'):
#                 # 1. Save temp
#                 if not os.path.exists("data"):
#                     os.makedirs("data")
#                 temp_path = os.path.join("data", "temp_upload.jpg")
#                 with open(temp_path, "wb") as f:
#                     f.write(uploaded_file.getbuffer())
                
#                 # 2. Predict
#                 label, confidence, extracted_text = predict_document(temp_path)
                
#                 # --- LABEL FIX (Optional: Keep this if you are using the generic model) ---
#                 label_map = {"LABEL_0": "Resume", "LABEL_1": "Email"}
#                 if label in label_map:
#                     label = label_map[label]
#                 # -------------------------------------------------------------------------

#                 # 3. Generate Summary (New Step!)
#                 # We generate this NOW so we can save it to the database immediately
#                 summary = generate_summary(extracted_text)

#                 # 4. Save to Database (Replaces 'save_and_log')
#                 # This saves the Image, Text, Summary, and Metadata into 'documind.db'
#                 db_msg = save_to_db(uploaded_file, label, confidence, extracted_text, summary)
#                 st.toast(db_msg, icon="🗄️")
                
#                 # 5. Save to Session State
#                 st.session_state['analyzed'] = True
#                 st.session_state['label'] = label
#                 st.session_state['confidence'] = confidence
#                 st.session_state['text'] = extracted_text
#                 st.session_state['summary'] = summary  # Save summary so we don't run it again

#         # if analyze_btn:
#         # --- RESULTS SECTION ---
#         if st.session_state.get('analyzed'):
#             st.markdown("---")
            
#             # 1. Classification Banner
#             st.subheader("1. Classification Result")
#             st.markdown(f"""
#             <div style="background-color: #3730A3; padding: 15px; border-radius: 10px; border-left: 5px solid #818CF8;">
#                 <h3 style="margin:0; color: white;">Category: {st.session_state['label'].upper()}</h3>
#                 <p style="margin:0; color: #C7D2FE;">Confidence Score: {st.session_state['confidence']:.2%}</p>
#             </div>
#             """, unsafe_allow_html=True)
            
#             st.write("") # Spacer

#             # 2. Text Stats (Cards)
#             st.subheader("📝 Text Metrics")
#             metrics = calculate_text_metrics(st.session_state['text'])
#             if metrics:
#                 c1, c2, c3, c4 = st.columns(4)
#                 c1.metric("Word Count", metrics["Word Count"])
#                 c2.metric("Sentences", metrics["Sentence Count"])
#                 c3.metric("Avg Word Len", metrics["Avg Word Length"])
#                 c4.metric("Readability", metrics["Readability Score (ARI)"])
            
#             st.write("") # Spacer
            
#             # --- DEFINE COLUMNS HERE ---
#             # This line must be indented exactly 12 spaces (inside the 'if analyzed' block)
#             col_left, col_right = st.columns(2)
            
#             # 3. Extraction (Left Column)
#             with col_left:
#                 st.subheader("2. Extracted Entities")
#                 details = extract_information(st.session_state['text'], st.session_state['label'])
#                 if details:
#                     st.table(pd.DataFrame(list(details.items()), columns=["Field", "Value"]))
#                 else:
#                     st.warning("No specific patterns found.")

#             # 4. Summarization (Right Column)
#             with col_right:
#                 st.subheader("3. AI Summary")
#                 if 'summary' in st.session_state:
#                     st.success(st.session_state['summary'])
#                 else:
#                     st.info("Summary available after analysis.")


#         #     with st.spinner('🔍 Scanning & Processing...'):
#         #         # 1. Save temp
#         #         if not os.path.exists("data"):
#         #             os.makedirs("data")
#         #         temp_path = os.path.join("data", "temp_upload.jpg")
#         #         with open(temp_path, "wb") as f:
#         #             f.write(uploaded_file.getbuffer())
                
#         #         # 2. Predict
#         #         label, confidence, extracted_text = predict_document(temp_path)
                
#         #         # 3. Archive
#         #         log_msg = save_and_log(uploaded_file, uploaded_file.name, label, confidence)
#         #         st.toast(log_msg, icon="✅")
                
#         #         # Save to Session State
#         #         st.session_state['analyzed'] = True
#         #         st.session_state['label'] = label
#         #         st.session_state['confidence'] = confidence
#         #         st.session_state['text'] = extracted_text
        
#         # --- RESULTS SECTION ---
#         # if st.session_state.get('analyzed'):
#         #     st.markdown("---")
            
#         #     # 1. Classification Banner (Full Width)
#         #     st.subheader("1. Classification Result")
#         #     # Custom HTML Banner for result
#         #     st.markdown(f"""
#         #     <div style="background-color: #3730A3; padding: 15px; border-radius: 10px; border-left: 5px solid #818CF8;">
#         #         <h3 style="margin:0; color: white;">Category: {st.session_state['label'].upper()}</h3>
#         #         <p style="margin:0; color: #C7D2FE;">Confidence Score: {st.session_state['confidence']:.2%}</p>
#         #     </div>
#         #     """, unsafe_allow_html=True)
            
#         #     st.write("") # Spacer

#         #     # 2. Text Stats (Cards)
#         #     st.subheader("📝 Text Metrics")
#         #     metrics = calculate_text_metrics(st.session_state['text'])
#         #     if metrics:
#         #         c1, c2, c3, c4 = st.columns(4)
#         #         c1.metric("Word Count", metrics["Word Count"])
#         #         c2.metric("Sentences", metrics["Sentence Count"])
#         #         c3.metric("Avg Word Len", metrics["Avg Word Length"])
#         #         c4.metric("Readability", metrics["Readability Score (ARI)"])
            
#         #     st.write("") # Spacer
            
#         #     col_left, col_right = st.columns(2)
            
#         #     with col_left:
#         #         # 3. Extraction
#         #         st.subheader("2. Extracted Entities")
#         #         details = extract_information(st.session_state['text'], st.session_state['label'])
#         #         if details:
#         #             st.table(pd.DataFrame(list(details.items()), columns=["Field", "Value"]))
#         #         else:
#         #             st.warning("No specific patterns found.")

#         #     # with col_right:
#         #     #     # 4. Summarization
#         #     #     st.subheader("3. AI Summary")
#         #     #     with st.spinner("Generating summary..."):
#         #     #         summary = generate_summary(st.session_state['text'])
#         #     #         st.success(summary)
#         #     # NEW CODE
#         # # This is likely inside the 'if st.session_state.get('analyzed'):' block
#         #     with col_right:
#         #         # 4. Summarization -> Make sure this line is indented
#         #         st.subheader("3. AI Summary")
            
#         #     # These lines must ALSO be indented to align with st.subheader
#         #         if 'summary' in st.session_state:
#         #             st.success(st.session_state['summary'])
#         #         else:
#         #             st.info("Summary available after analysis.")


#             # 5. Visual Analytics
#             st.subheader("4. Context Cloud")
#             try:
#                 # Dark mode compatible wordcloud
#                 wordcloud = WordCloud(width=1000, height=400, background_color='#1F2937', colormap='cool').generate(st.session_state['text'])
#                 fig, ax = plt.subplots(figsize=(10, 4))
#                 # Make plot background transparent to match theme
#                 fig.patch.set_alpha(0)
#                 ax.imshow(wordcloud, interpolation='bilinear')
#                 ax.axis("off")
#                 st.pyplot(fig)
#             except:
#                 st.write("Not enough text for visualization.")

# # ==========================================
# # PAGE 2: HISTORY LOG
# # ==========================================
# # NEW CODE
# elif page == "History Log":
#     st.title("📜 Database History")
#     st.write("View all archived documents stored in SQLite.")
    
#     # 1. Get Data from DB
#     df_history = get_db_history()
    
#     if not df_history.empty:
#         # 2. Display Data
#         st.dataframe(
#             df_history,
#             column_config={
#                 "upload_date": st.column_config.DatetimeColumn("Upload Date", format="D MMM YYYY, h:mm a"),
#                 "filename": st.column_config.TextColumn("File Name"),
#                 "category": st.column_config.TextColumn("Category"),
#                 "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1.0, format="%.2f"),
#                 "summary": st.column_config.TextColumn("Summary", width="large"),
#             },
#             use_container_width=True,
#             hide_index=True
#         )
#     else:
#         st.info("No documents found in the database yet.")



# elif page == "History Log":
#     st.title("📜 Process History")
#     st.write("Manage your archived document logs below.")
    
#     df_history = get_history()
    
#     if not df_history.empty:
#         df_history.insert(0, "Select", False)
        
#         edited_df = st.data_editor(
#             df_history,
#             column_config={
#                 "Select": st.column_config.CheckboxColumn("Delete?", help="Select rows to remove", width="small"),
#                 "Date": st.column_config.DatetimeColumn("Date", format="D MMM YYYY, h:mm a"),
#                 "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%f%%"),
#                 "Filename": st.column_config.TextColumn("Filename"),
#                 "Predicted_Category": st.column_config.TextColumn("Category"),
#             },
#             disabled=["Date", "Filename", "Predicted_Category", "Confidence", "Saved_Path"],
#             use_container_width=True,
#             hide_index=True,
#             height=600
#         )
        
#         rows_to_delete = edited_df[edited_df.Select].index

        
#         if len(rows_to_delete) > 0:
#             st.warning(f"⚠️ You have selected {len(rows_to_delete)} document(s) to delete.")
            
#             if st.button("🗑️ Delete Selected", type="primary"):
#                 # Get the actual Database IDs (hidden in the dataframe)
#                 ids_to_delete = edited_df.loc[rows_to_delete, "id"].tolist()
                
#                 success = delete_db_entries(ids_to_delete)
#                 if success:
#                     st.success("Entries deleted successfully!")
#                     st.rerun()
#                 else:
#                     st.error("Error deleting entries.")
        
#         # if len(rows_to_delete) > 0:
#         #     st.warning(f"⚠️ You have selected {len(rows_to_delete)} document(s) to delete.")
#         #     if st.button("🗑️ Delete Selected", type="primary"):
#         #         success = delete_history_entries(rows_to_delete)
#         #         if success:
#         #             st.success("Entries deleted successfully!")
#         #             st.rerun()
#         #         else:
#         #             st.error("Error deleting entries.")
#     else:
#         st.info("No documents processed yet.")


# # ==========================================
# # PAGE 3: ANALYTICS
# # ==========================================
# # NEW CODE
# elif page == "System Analytics":
#     st.title("📊 System Analytics")
    
#     df_history = get_db_history()
    
#     if not df_history.empty:
#         colA, colB = st.columns(2)
        
#         with colA:
#             st.subheader("Uploads by Category")
#             # Database column is 'category'
#             st.bar_chart(df_history['category'].value_counts())
            
#         with colB:
#             st.subheader("AI Confidence Trend")
#             try:
#                 # Database stores confidence as a float (0.95), not string ("95%")
#                 # So we don't need to strip '%' anymore!
#                 st.line_chart(df_history['confidence'])
#             except:
#                 st.write("Insufficient data for trend analysis.")
#     else:
#         st.info("No data available. Process some documents first!")


# elif page == "System Analytics":
#     st.title("📊 System Analytics")
    
#     df_history = get_history()
#     if not df_history.empty:
#         colA, colB = st.columns(2)
        
#         with colA:
#             st.subheader("Uploads by Category")
#             st.bar_chart(df_history['Predicted_Category'].value_counts())
            
#         with colB:
#             st.subheader("AI Confidence Trend")
#             try:
#                 df_history['Conf_Value'] = df_history['Confidence'].str.rstrip('%').astype('float')
#                 st.line_chart(df_history['Conf_Value'])
#             except:
#                 st.write("Insufficient data for trend analysis.")
#     else:
#         st.info("No data available. Process some documents first!")









