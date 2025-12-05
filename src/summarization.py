from transformers import pipeline

# Load a lighter summarization model (DistilBART)
# This downloads automatically the first time you run it.
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def generate_summary(text):
    """
    Summarizes long document text into a short paragraph.
    """
    # 1. Safety Check: If text is too short, don't summarize
    if len(text.split()) < 50:
        return "Document is too short to summarize."

    # 2. Chunking (Handling long documents)
    # Models have a limit (usually 1024 tokens). We take the first ~3000 chars.
    # For a portfolio project, summarizing the first page is usually enough.
    max_input_length = 3000 
    header_hint = text[:100] 
    input_text = header_hint + "\n" + text[:max_input_length]

    try:
        # 3. Generate Summary
        # min_length=30 ensures it's not too brief
        # max_length=130 ensures it's concise
        summary_output = summarizer(input_text, max_length=130, min_length=30, do_sample=False)
        
        return summary_output[0]['summary_text']
    
    except Exception as e:
        return f"Error generating summary: {str(e)}"