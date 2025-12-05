import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

# CONFIG
MODEL_PATH = os.path.join("models", "documind_v1")
DATA_PATH = os.path.join("data", "processed", "documind_dataset.csv")

def evaluate():
    print("⏳ Loading Model & Test Data...")
    
    # 1. Load Model
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    except:
        print("❌ Model not found! Train it first.")
        return

    # 2. Load Data
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['text'])
    
    # Create Labels
    label_list = df['category'].unique().tolist()
    label_list.sort()
    label2id = {label: i for i, label in enumerate(label_list)}
    
    # Filter for TEST split only (Data the model hasn't seen)
    test_df = df[df['split'] == 'test']
    
    # Optimization: Use a smaller sample for speed if dataset is huge
    if len(test_df) > 500:
        test_df = test_df.sample(500, random_state=42)
        
    print(f"✅ Evaluating on {len(test_df)} test documents...")

    # 3. Run Predictions
    predictions = []
    true_labels = []

    model.eval() # Set to evaluation mode
    
    print("🚀 Running Inference...")
    for index, row in test_df.iterrows():
        text = row['text']
        true_label_id = label2id[row['category']]
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=1).item()
            
        predictions.append(pred_id)
        true_labels.append(true_label_id)

    # 4. Calculate Metrics
    acc = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=label_list, output_dict=True)
    macro_f1 = report['macro avg']['f1-score']

    print("\n" + "="*30)
    print("📊 FINAL EVALUATION REPORT")
    print("="*30)
    print(f"✅ Accuracy:  {acc:.2%}")
    print(f"✅ Macro F1:  {macro_f1:.2f}")
    print("-" * 30)

    # 5. Generate Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Plot it
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_list, yticklabels=label_list)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Save the plot
    save_path = os.path.join("data", "confusion_matrix.png")
    plt.savefig(save_path)
    print(f"📉 Confusion Matrix saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    evaluate()