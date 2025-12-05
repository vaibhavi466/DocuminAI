# gemini version
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pytesseract
from PIL import Image
import os

# CONFIG
# We load the model from the folder where training will save it
MODEL_DIR = os.path.join("models", "documind_v1") 

# Tesseract Path (Keep your existing path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def predict_document(image_path):
    """
    1. Reads the image.
    2. Extracts text using OCR.
    3. Feeds text to DistilBERT.
    4. Returns the category.
    """
    
    # 1. OCR: Get text from image
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
    except Exception as e:
        return f"Error reading image: {e}", 0.0

    # If OCR failed to find text
    if not text.strip():
        return "No text found in document", 0.0

    # 2. Load Model (Only if not already loaded)
    # We do this inside the function or globally
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    except OSError:
        return "Model not found. Wait for training to finish!", 0.0

    # 3. Prepare Text for AI
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=512
    )

    # 4. Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get the highest probability
        confidence, predicted_class_idx = torch.max(probs, dim=-1)
        
        # Get the label name (e.g., "invoice")
        label = model.config.id2label[predicted_class_idx.item()]
    
    return label, confidence.item(), text

if __name__ == "__main__":
    # Test with a dummy path
    print("Inference script ready. Run via App.")




#  chatgpt version
# import os
# import io
# import torch
# import numpy as np
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import pytesseract
# from PIL import Image
# import cv2

# # CONFIG
# MODEL_DIR = os.path.join("models", "documind_v1")
# # Tesseract Path (keep your existing path)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # Globals for caching
# _tokenizer = None
# _model = None
# _device = torch.device("cpu")

# # Logging folder for OCR outputs (helps collect misclassified samples)
# LOG_DIR = "ocr_logs"
# os.makedirs(LOG_DIR, exist_ok=True)


# def _load_model_and_tokenizer():
#     global _tokenizer, _model, _device
#     if _tokenizer is None or _model is None:
#         try:
#             _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
#             _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
#             _model.to(_device)
#             _model.eval()
#         except OSError:
#             raise RuntimeError("Model not found. Wait for training to finish or check MODEL_DIR.")
#     return _tokenizer, _model


# def _preprocess_image_for_ocr(image_path):
#     """Read image -> grayscale -> denoise -> threshold -> return PIL Image"""
#     image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
#     if image is None:
#         raise ValueError(f"Unable to read image: {image_path}")

#     # Convert to gray
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Resize if very small or large (keeps aspect ratio)
#     h, w = gray.shape
#     target_width = 1600
#     if w < target_width and w > 0:
#         scale = target_width / float(w)
#         gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

#     # Denoise
#     gray = cv2.medianBlur(gray, 3)

#     # Adaptive thresholding often helps OCR
#     thresh = cv2.adaptiveThreshold(
#         gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
#     )

#     # Optionally deskew: compute moments / rotate (skipped for simplicity)

#     # Convert back to PIL Image (pytesseract prefers PIL)
#     pil_img = Image.fromarray(thresh)
#     return pil_img


# def _ocr_image(image_path, psm=6, oem=3):
#     """
#     Run pytesseract OCR with configurable settings.
#     psm: Page segmentation mode. 6 = Assume a single uniform block of text.
#     oem: OCR Engine Mode. 3 = Default (both LSTM and legacy).
#     """
#     pil_img = _preprocess_image_for_ocr(image_path)
#     custom_config = f"--oem {oem} --psm {psm}"
#     text = pytesseract.image_to_string(pil_img, config=custom_config)
#     return text


# def _keyword_email_check(text):
#     """Return True if text contains strong email cues."""
#     email_markers = ["from:", "to:", "subject:", "sent:", "cc:", "bcc:", "dear ", "regards", "sincerely"]
#     text_l = text.lower()
#     match_count = sum(1 for m in email_markers if m in text_l)
#     # heuristics: if 2 or more email markers are present, treat as email
#     return match_count >= 2


# def _chunk_text(text, max_chars=3000):
#     """Simple chunk-by-character splitter (keeps order)."""
#     if len(text) <= max_chars:
#         return [text]
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = min(start + max_chars, len(text))
#         chunks.append(text[start:end])
#         start = end
#     return chunks


# def predict_document(image_path, top_k=3, override_threshold=0.90):
#     """
#     Returns: (predicted_label, confidence, top_k_list, ocr_text)
#       - predicted_label: str label
#       - confidence: float (0..1)
#       - top_k_list: list of tuples (label, prob)
#       - ocr_text: the OCR'd text (string) saved for debugging
#     """
#     # 1. OCR
#     try:
#         text = _ocr_image(image_path)
#     except Exception as e:
#         return "Error reading image", 0.0, [], f"OCR error: {e}"

#     # Save OCR text for debugging (helps identify why misclassified)
#     base_name = os.path.splitext(os.path.basename(image_path))[0]
#     log_file = os.path.join(LOG_DIR, f"{base_name}_ocr.txt")
#     with open(log_file, "w", encoding="utf-8") as f:
#         f.write(text)

#     if not text.strip():
#         return "No text found in document", 0.0, [], text

#     # Quick rule-based override for emails (cheap and effective)
#     try:
#         if _keyword_email_check(text):
#             # Return email with high pseudo-confidence but still let model exist
#             return "email", 0.99, [("email", 0.99)], text
#     except Exception:
#         pass  # don't fail on heuristic

#     # 2. Load model & tokenizer
#     try:
#         tokenizer, model = _load_model_and_tokenizer()
#     except RuntimeError as e:
#         return str(e), 0.0, [], text

#     # 3. Chunking to avoid losing text due to truncation
#     chunks = _chunk_text(text, max_chars=3000)  # adjust max_chars as needed
#     all_probs = None

#     with torch.no_grad():
#         for chunk in chunks:
#             inputs = tokenizer(
#                 chunk,
#                 return_tensors="pt",
#                 truncation=True,
#                 padding="max_length",
#                 max_length=512
#             )
#             inputs = {k: v.to(_device) for k, v in inputs.items()}
#             outputs = model(**inputs)
#             logits = outputs.logits
#             probs = torch.nn.functional.softmax(logits, dim=-1)  # shape (1, num_labels)
#             probs = probs.cpu().numpy()
#             if all_probs is None:
#                 all_probs = probs
#             else:
#                 all_probs += probs

#     # Average probs across chunks
#     avg_probs = all_probs / len(chunks)
#     avg_probs = avg_probs.flatten()  # numpy array

#     # Top-k and predicted class
#     topk_idx = np.argsort(avg_probs)[::-1][:top_k]
#     id2label = model.config.id2label
#     topk = [(id2label[int(i)], float(avg_probs[int(i)])) for i in topk_idx]
#     predicted_label, confidence = topk[0]

#     # Optional: if top confidence is not high enough, mark uncertain
#     if confidence < override_threshold:
#         # you can choose to return 'uncertain' or still return predicted_label
#         # For now return predicted label but include top_k for debugging
#         pass

#     return predicted_label, float(confidence), topk, text


# if __name__ == "__main__":
#     # quick local test: replace with an actual path
#     test_path = "example_email_screenshot.png"
#     if os.path.exists(test_path):
#         label, conf, topk, ocr = predict_document(test_path)
#         print("Label:", label, "Conf:", conf)
#         print("Top-k:", topk)
#         print("OCR sample:", ocr[:400])
#     else:
#         print("Inference script ready. Call predict_document(image_path).")
