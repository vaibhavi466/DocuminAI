import os
import pandas as pd
from ocr_engine import extract_text

# Define where our data lives
DATA_DIR = os.path.join("data", "raw")
OUTPUT_FILE = os.path.join("data", "processed", "documind_dataset.csv")

def create_dataset():
    """
    Loops through train/val/test folders, reads images, 
    extracts text, and saves to a CSV.
    """
    data = []
    
    # We will loop through 'train', 'val', and 'test'
    splits = ['train', 'val', 'test']
    
    print("🚀 Starting Dataset Creation... this might take a while!")
    
    for split in splits:
        split_path = os.path.join(DATA_DIR, split)
        
        # Check if the split folder exists (e.g. data/raw/train)
        if not os.path.exists(split_path):
            print(f"⚠️ Warning: {split_path} not found. Skipping.")
            continue
            
        # Loop through categories (invoice, email, etc.)
        categories = os.listdir(split_path)
        
        for category in categories:
            category_path = os.path.join(split_path, category)
            
            # Skip if it's a file, we only want folders
            if not os.path.isdir(category_path):
                continue
                
            print(f"📂 Processing {split} / {category} ...")
            
            # Get all images in this category
            images = os.listdir(category_path)
            
            # --- LIMIT FOR TESTING ---
            # processing 1000s of images takes hours. 
            # Let's process only 5 per category to test the code first.
            # Change this to 'images' (remove [:5]) later to run on ALL files.
            images_to_process = images[:100]
            
            for img_name in images_to_process:
                img_path = os.path.join(category_path, img_name)
                
                # 1. Run OCR
                extracted_text = extract_text(img_path)
                
                # 2. Add to list if text was found
                if extracted_text:
                    data.append({
                        "filename": img_name,
                        "category": category,
                        "split": split,
                        "text": extracted_text.strip() # Remove extra spaces
                    })
    
    # 3. Save to CSV
    print(f"✅ Processing complete! Found {len(data)} documents.")
    
    if len(data) > 0:
        df = pd.DataFrame(data)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"💾 Dataset saved to: {OUTPUT_FILE}")
        
        # Show a sneak peek
        print("\n--- First 5 Rows ---")
        print(df.head())
    else:
        print("❌ No data extracted. Check your paths.")

if __name__ == "__main__":
    create_dataset()