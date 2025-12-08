from transformers import pipeline

# Load a lighter summarization model (DistilBART) - NO CHANGE TO MODEL
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def generate_summary(text):
    """
    Summarizes long document text into a short paragraph.
    Includes text cleaning for better OCR processing.
    """
    
    # 1. ROBUST TEXT CLEANING AND PRE-PROCESSING
    
    # Replace common OCR noise: newlines, multiple spaces, and non-essential chars
    cleaned_text = text.replace('\n', ' ')
    cleaned_text = cleaned_text.replace('\r', ' ')
    cleaned_text = cleaned_text.replace('\t', ' ')
    
    # Collapse multiple spaces into a single space
    cleaned_text = ' '.join(cleaned_text.split())
    
    # 2. SAFETY CHECK
    
    # Use the cleaned text length for the check
    if len(cleaned_text.split()) < 50:
        return "Document is too short to summarize (less than 50 words)."

    # 3. ACCURATE TOKEN CHUNKING
    
    # The model has a 1024 token limit. We use the tokenizer to accurately chunk the input.
    max_token_limit = 1000 

    try:
        # Encode the clean text, automatically truncate if longer than the limit
        input_ids = summarizer.tokenizer.encode(
            cleaned_text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=max_token_limit
        )
        
        # Decode the chunked tokens back into a string for the pipeline
        input_text = summarizer.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        # 4. GENERATE SUMMARY (with adjusted length parameters for slightly longer output)
        
        # Increased max_length from 130 to 180 and min_length from 30 to 50
        summary_output = summarizer(
            input_text, 
            max_length=180, 
            min_length=50, 
            do_sample=False,
            # Added a hint to improve coherence
            # Note: This parameter is model-specific and might not exist on all models
            # but is sometimes supported by the pipeline API.
            # early_stopping=True 
        )
        
        # 5. CLEAN THE OUTPUT (Removing leading/trailing spaces/newlines from model output)
        final_summary = summary_output[0]['summary_text'].strip()
        
        return final_summary
    
    except Exception as e:
        # Improved error message for debugging
        return f"Error generating summary after cleaning: {type(e).__name__}: {str(e)}"
    
    

# from transformers import pipeline

# # Load a lighter summarization model (DistilBART)
# # This downloads automatically the first time you run it.
# summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# def generate_summary(text):
#     """
#     Summarizes long document text into a short paragraph.
#     """
#     # 1. Safety Check: If text is too short, don't summarize
#     if len(text.split()) < 50:
#         return "Document is too short to summarize."

#     # 2. Chunking (Handling long documents)
#     # Models have a limit (usually 1024 tokens). We take the first ~3000 chars.
#     # For a portfolio project, summarizing the first page is usually enough.
#     max_input_length = 3000 
#     header_hint = text[:100] 
#     input_text = header_hint + "\n" + text[:max_input_length]

#     try:
#         # 3. Generate Summary
#         # min_length=30 ensures it's not too brief
#         # max_length=130 ensures it's concise
#         summary_output = summarizer(input_text, max_length=130, min_length=30, do_sample=False)
        
#         return summary_output[0]['summary_text']
    
#     except Exception as e:
#         return f"Error generating summary: {str(e)}"