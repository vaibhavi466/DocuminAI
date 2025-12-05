import mlflow
import os
import pandas as pd
import torch
import numpy as np
os.environ["MLFLOW_EXPERIMENT_NAME"] = "DocuMind_Experiments"
os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "TRUE"
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset, ClassLabel

# 1. SETUP CONFIGURATION
# ======================
# We switch to DistilBERT, which is lighter and faster for CPU training
MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = os.path.join("data", "processed", "documind_dataset.csv")
OUTPUT_DIR = os.path.join("models", "documind_v1")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training on: {device.upper()}")

    # 1. Load Data
    print("⏳ Loading Dataset...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ File not found: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    
    # Remove empty rows
    df = df.dropna(subset=['text'])
    
    # Create Labels
    label_list = df['category'].unique().tolist()
    label_list.sort() # Ensure consistent order
    num_labels = len(label_list)
    
    # Create id2label mappings
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    
    print(f"✅ Categories found: {label_list}")
    
    # Map text categories to numbers in the dataframe
    df['label'] = df['category'].map(label2id)
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    
    # Split: 80% Train, 20% Test
    dataset = dataset.train_test_split(test_size=0.2)
    
    # 2. Tokenization
    print(f"⬇️ Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def preprocess_function(examples):
        # DistilBERT only needs the text, no bounding boxes!
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    print("⚙️ Tokenizing data...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # 3. Model Setup
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    
    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # 6. Train
    print("🔥 Starting Training...")
    trainer.train()
    
    # 7. Save
    print("💾 Saving Final Model...")
    trainer.save_model(OUTPUT_DIR)
    print(f"🎉 Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()