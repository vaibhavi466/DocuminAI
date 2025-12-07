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
from src.utils import save_and_log, get_history, calculate_text_metrics, delete_history_entries

# 1. Page Config
st.set_page_config(page_title="DocuMind AI", page_icon="📄", layout="wide")

# 2. Custom CSS for "Neon/Dark" Look
def load_css():
    st.markdown("""
        <style>
        /* Global Font and Background adjustments */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #111827; /* Darker sidebar */
            border-right: 1px solid #374151;
        }

        /* Card Styling for Metrics */
        div[data-testid="stMetric"] {
            background-color: #1F2937; /* Dark card background */
            border: 1px solid #374151;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        div[data-testid="stMetric"] label {
            color: #9CA3AF !important; /* Muted text for labels */
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #F3F4F6 !important; /* Bright text for values */
        }

        /* Custom Button Styling - "Neon" primary button */
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
            border-color: #8B5CF6; /* Neon purple on hover */
        }

        /* Headers and Titles */
        h1, h2, h3 {
            color: #F3F4F6 !important;
        }
        
        /* Table Styling */
        div[data-testid="stTable"] {
            color: #E5E7EB;
        }
        </style>
    """, unsafe_allow_html=True)

load_css()

# 3. Sidebar Navigation
with st.sidebar:
    # You can add your Logo here if you have one
    # st.image("logo.png", width=200) 
    st.title("🧩 DocuMind AI")
    st.markdown("---")
    page = st.radio("Navigate to:", ["Analysis Dashboard", "History Log", "System Analytics"])
    st.markdown("---")
    st.caption("v1.0 | Powered by LayoutLM & SpaCy")

# ==========================================
# PAGE 1: ANALYSIS DASHBOARD
# ==========================================
if page == "Analysis Dashboard":
    # Header Section with a gradient text effect (optional HTML hack)
    st.markdown("""
    <h1 style='text-align: center; background: -webkit-linear-gradient(45deg, #60A5FA, #A78BFA); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
    Intelligent Document Analysis
    </h1>
    """, unsafe_allow_html=True)
    
    st.write("") # Spacer

    # --- UPLOAD SECTION ---
    st.container()
    uploaded_file = st.file_uploader("📂 Upload Document (JPG, PNG, TIF)", type=["jpg", "png", "tif", "jpeg"])

    if uploaded_file is not None:
        st.markdown("---")
        # Layout: Image on Left, Actions on Right
        col1, col2 = st.columns([1, 2])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption='Document Preview', use_container_width=True)
            # NEW CODE (Fixes the warning)
            
        with col2:
            st.subheader("Processing Options")
            st.info("Document detected. The system will now extract text, classify the document type, and generate a summary.")
            
            # Primary Action Button
            analyze_btn = st.button('🚀 Analyze & Archive Document', use_container_width=True, type="primary")

        # Logic
        if analyze_btn:
            with st.spinner('🔍 Scanning & Processing...'):
                # 1. Save temp
                if not os.path.exists("data"):
                    os.makedirs("data")
                temp_path = os.path.join("data", "temp_upload.jpg")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 2. Predict
                label, confidence, extracted_text = predict_document(temp_path)
                
                # 3. Archive
                log_msg = save_and_log(uploaded_file, uploaded_file.name, label, confidence)
                st.toast(log_msg, icon="✅")
                
                # Save to Session State
                st.session_state['analyzed'] = True
                st.session_state['label'] = label
                st.session_state['confidence'] = confidence
                st.session_state['text'] = extracted_text
        
        # --- RESULTS SECTION ---
        if st.session_state.get('analyzed'):
            st.markdown("---")
            
            # 1. Classification Banner (Full Width)
            st.subheader("1. Classification Result")
            # Custom HTML Banner for result
            st.markdown(f"""
            <div style="background-color: #3730A3; padding: 15px; border-radius: 10px; border-left: 5px solid #818CF8;">
                <h3 style="margin:0; color: white;">Category: {st.session_state['label'].upper()}</h3>
                <p style="margin:0; color: #C7D2FE;">Confidence Score: {st.session_state['confidence']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("") # Spacer

            # 2. Text Stats (Cards)
            st.subheader("📝 Text Metrics")
            metrics = calculate_text_metrics(st.session_state['text'])
            if metrics:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Word Count", metrics["Word Count"])
                c2.metric("Sentences", metrics["Sentence Count"])
                c3.metric("Avg Word Len", metrics["Avg Word Length"])
                c4.metric("Readability", metrics["Readability Score (ARI)"])
            
            st.write("") # Spacer
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                # 3. Extraction
                st.subheader("2. Extracted Entities")
                details = extract_information(st.session_state['text'], st.session_state['label'])
                if details:
                    st.table(pd.DataFrame(list(details.items()), columns=["Field", "Value"]))
                else:
                    st.warning("No specific patterns found.")

            with col_right:
                # 4. Summarization
                st.subheader("3. AI Summary")
                with st.spinner("Generating summary..."):
                    summary = generate_summary(st.session_state['text'])
                    st.success(summary)

            # 5. Visual Analytics
            st.subheader("4. Context Cloud")
            try:
                # Dark mode compatible wordcloud
                wordcloud = WordCloud(width=1000, height=400, background_color='#1F2937', colormap='cool').generate(st.session_state['text'])
                fig, ax = plt.subplots(figsize=(10, 4))
                # Make plot background transparent to match theme
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
    st.title("📜 Process History")
    st.write("Manage your archived document logs below.")
    
    df_history = get_history()
    
    if not df_history.empty:
        df_history.insert(0, "Select", False)
        
        edited_df = st.data_editor(
            df_history,
            column_config={
                "Select": st.column_config.CheckboxColumn("Delete?", help="Select rows to remove", width="small"),
                "Date": st.column_config.DatetimeColumn("Date", format="D MMM YYYY, h:mm a"),
                "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%f%%"),
                "Filename": st.column_config.TextColumn("Filename"),
                "Predicted_Category": st.column_config.TextColumn("Category"),
            },
            disabled=["Date", "Filename", "Predicted_Category", "Confidence", "Saved_Path"],
            use_container_width=True,
            hide_index=True,
            height=600
        )
        
        rows_to_delete = edited_df[edited_df.Select].index
        
        if len(rows_to_delete) > 0:
            st.warning(f"⚠️ You have selected {len(rows_to_delete)} document(s) to delete.")
            if st.button("🗑️ Delete Selected", type="primary"):
                success = delete_history_entries(rows_to_delete)
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
    
    df_history = get_history()
    if not df_history.empty:
        colA, colB = st.columns(2)
        
        with colA:
            st.subheader("Uploads by Category")
            st.bar_chart(df_history['Predicted_Category'].value_counts())
            
        with colB:
            st.subheader("AI Confidence Trend")
            try:
                df_history['Conf_Value'] = df_history['Confidence'].str.rstrip('%').astype('float')
                st.line_chart(df_history['Conf_Value'])
            except:
                st.write("Insufficient data for trend analysis.")
    else:
        st.info("No data available. Process some documents first!")


# import streamlit as st
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# from PIL import Image
# from wordcloud import WordCloud
# import pandas as pd

# # Import our custom modules
# from src.inference import predict_document
# from src.extraction import extract_information
# from src.summarization import generate_summary
# from src.utils import save_and_log, get_history, calculate_text_metrics, delete_history_entries

# # 1. Page Config
# st.set_page_config(page_title="DocuMind AI", page_icon="📄", layout="wide")

# # 2. Custom CSS for "Cards" look
# def load_css():
#     st.markdown("""
#         <style>
#         div[data-testid="stMetric"] {
#             background-color: #F8FAFC;
#             border: 1px solid #E2E8F0;
#             padding: 15px;
#             border-radius: 8px;
#             box-shadow: 0 2px 4px rgba(0,0,0,0.05);
#         }
#         </style>
#     """, unsafe_allow_html=True)
# load_css()

# # 3. Sidebar Navigation
# with st.sidebar:
#     st.title("🧩 Menu")
#     page = st.radio("Navigate to:", ["Analysis Dashboard", "History Log", "System Analytics"])
#     st.markdown("---")
#     st.caption("DocuMind AI v1.0")

# # ==========================================
# # PAGE 1: ANALYSIS DASHBOARD (The "Front Page")
# # ==========================================
# if page == "Analysis Dashboard":
#     st.title("📄 DocuMind AI")
#     st.markdown("### Intelligent Document Analysis System")
    
#     # --- UPLOAD SECTION (Now in Main Area) ---
#     st.container()
#     uploaded_file = st.file_uploader("📂 Upload your document here (JPG, PNG, TIF)", type=["jpg", "png", "tif", "jpeg"])

#     if uploaded_file is not None:
#         # Layout: Image on Left, Actions on Right
#         col1, col2 = st.columns([1, 2])
        
#         with col1:
#             image = Image.open(uploaded_file)
#             st.image(image, caption='Preview', use_column_width=True, width=300)
            
#         with col2:
#             st.write("Document detected. Ready to process.")
#             analyze_btn = st.button('🚀 Analyze & Archive Document', use_container_width=True, type="primary")

#         # Logic
#         if analyze_btn:
#             with st.spinner('Processing OCR & AI Models...'):
#                 # 1. Save temp
#                 temp_path = os.path.join("data", "temp_upload.jpg")
#                 with open(temp_path, "wb") as f:
#                     f.write(uploaded_file.getbuffer())
                
#                 # 2. Predict
#                 label, confidence, extracted_text = predict_document(temp_path)
                
#                 # 3. Archive
#                 log_msg = save_and_log(uploaded_file, uploaded_file.name, label, confidence)
#                 st.toast(log_msg, icon="✅")
                
#                 # Save to Session State (so it doesn't disappear)
#                 st.session_state['analyzed'] = True
#                 st.session_state['label'] = label
#                 st.session_state['confidence'] = confidence
#                 st.session_state['text'] = extracted_text
        
#         # --- RESULTS SECTION ---
#         if st.session_state.get('analyzed'):
#             st.markdown("---")
            
#             # 1. Classification Banner
#             st.subheader("1. Classification Result")
#             st.info(f"**Category: {st.session_state['label'].upper()}** (Confidence: {st.session_state['confidence']:.2%})")
            
#             # 2. Text Stats (Cards)
#             st.subheader("📝 Text Analysis")
#             metrics = calculate_text_metrics(st.session_state['text'])
#             if metrics:
#                 c1, c2, c3, c4 = st.columns(4)
#                 c1.metric("Word Count", metrics["Word Count"])
#                 c2.metric("Sentences", metrics["Sentence Count"])
#                 c3.metric("Avg Word Len", metrics["Avg Word Length"])
#                 c4.metric("Readability", metrics["Readability Score (ARI)"])
            
#             col_left, col_right = st.columns(2)
            
#             with col_left:
#                 # 3. Extraction
#                 st.subheader("2. Key Information")
#                 details = extract_information(st.session_state['text'], st.session_state['label'])
#                 if details:
#                     # Display as a clean table
#                     st.table(pd.DataFrame(list(details.items()), columns=["Field", "Value"]))
#                 else:
#                     st.warning("No specific patterns found.")

#             with col_right:
#                 # 4. Summarization
#                 st.subheader("3. Content Summary")
#                 with st.spinner("Summarizing..."):
#                     summary = generate_summary(st.session_state['text'])
#                     st.success(summary)

#             # 5. Visual Analytics (Full Width)
#             st.subheader("4. Visual Analytics")
#             try:
#                 wordcloud = WordCloud(width=1000, height=400, background_color='white').generate(st.session_state['text'])
#                 fig, ax = plt.subplots(figsize=(10, 4))
#                 ax.imshow(wordcloud, interpolation='bilinear')
#                 ax.axis("off")
#                 st.pyplot(fig)
#             except:
#                 st.write("Not enough text for visualization.")

# # ==========================================
# # PAGE 2: HISTORY LOG (UPDATED WITH DELETE)
# # ==========================================
# elif page == "History Log":
#     st.title("📜 Process History")
#     st.write("Manage your archived document logs below.")
    
#     df_history = get_history()
    
#     if not df_history.empty:
#         # Add a checkbox column for selection
#         df_history.insert(0, "Select", False)
        
#         # Use Data Editor instead of Dataframe to allow interaction
#         edited_df = st.data_editor(
#             df_history,
#             column_config={
#                 "Select": st.column_config.CheckboxColumn("Delete?", help="Select rows to remove", width="small"),
#                 "Date": st.column_config.DatetimeColumn("Date", format="D MMM YYYY, h:mm a"),
#                 "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%f%%"),
#                 "Filename": st.column_config.TextColumn("Filename"),
#                 "Predicted_Category": st.column_config.TextColumn("Category"),
#             },
#             disabled=["Date", "Filename", "Predicted_Category", "Confidence", "Saved_Path"], # Prevent editing text
#             use_container_width=True,
#             hide_index=True,
#             height=600
#         )
        
#         # Delete Logic
#         rows_to_delete = edited_df[edited_df.Select].index
        
#         if len(rows_to_delete) > 0:
#             st.warning(f"⚠️ You have selected {len(rows_to_delete)} document(s) to delete.")
            
#             col_del, col_space = st.columns([1, 5])
#             with col_del:
#                 if st.button("🗑️ Delete Selected", type="primary"):
#                     success = delete_history_entries(rows_to_delete)
#                     if success:
#                         st.success("Entries deleted successfully!")
#                         st.rerun()
#                     else:
#                         st.error("Error deleting entries.")
        
#     else:
#         st.info("No documents processed yet.")

# # ==========================================
# # PAGE 3: ANALYTICS
# # ==========================================
# elif page == "System Analytics":
#     st.title("📊 System Analytics")
    
#     df_history = get_history()
#     if not df_history.empty:
#         colA, colB = st.columns(2)
        
#         with colA:
#             st.subheader("Uploads by Category")
#             # Using Streamlit's native interactive charts
#             st.bar_chart(df_history['Predicted_Category'].value_counts())
            
#         with colB:
#             st.subheader("AI Confidence Trend")
#             try:
#                 # Clean percentage string to float
#                 df_history['Conf_Value'] = df_history['Confidence'].str.rstrip('%').astype('float')
#                 # Line chart shows performance over time
#                 st.line_chart(df_history['Conf_Value'])
#             except:
#                 st.write("Insufficient data for trend analysis.")
#     else:
#         st.info("No data available. Process some documents first!")