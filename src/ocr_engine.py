import pytesseract
from PIL import Image # Python Imaging Library
import os

# ==========================================
# WINDOWS CONFIGURATION (THE MAGIC LINE)
# ==========================================
# IMPORTANT: Check if this path matches where you installed Tesseract!
# Common paths:
# 1. C:\Program Files\Tesseract-OCR\tesseract.exe
# 2. C:\Program Files (x86)\Tesseract-OCR\tesseract.exe
# 3. C:\Users\YourName\AppData\Local\Tesseract-OCR\tesseract.exe

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ==========================================

def extract_text(image_path):
    """
    Reads an image from the path and returns the text string.
    """
    try:
        # 1. Load the image
        img = Image.open(image_path)
        
        # 2. Convert to text using Tesseract
        # custom_config allows us to handle simple layouts better
        text = pytesseract.image_to_string(img)
        
        return text
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return ""

# ==========================================
# TEST BLOCK (This only runs if you run this file directly)
# ==========================================
if __name__ == "__main__":
    # Let's test it on a dummy file. 
    # WE need to find a real file in your 'train' folder to test.
    
    # Update this path to point to a REAL file in your data folder
    test_image_path = os.path.join("data", "raw", "train", "invoice" , "0000145869.tif")
    
    print(f"Testing OCR on: {test_image_path}")
    
    # Run the function
    result = extract_text(test_image_path)
    
    print("-" * 30)
    print("EXTRACTED TEXT:")
    print("-" * 30)
    print(result)