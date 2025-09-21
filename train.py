import os
import sys
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
import evaluate

# -----------------------
# CONFIG
# -----------------------
MODEL_NAME = "FacebookAI/xlm-roberta-base"
DATA_DIR = "data"   # ต้องมี train_en_sentiment.csv, val_en_sentiment.csv, test_en_sentiment.csv
TEXT_COL = "text"
LABEL_COL = "label"
NUM_LABELS = 3
OUTPUT_DIR = "./xlmr-finetuned-course-sentiment"
MAX_LEN = 256
SEED = 42
# -----------------------

def safe_check_files():
    missing = []
    for fn in ["train_en_sentiment.csv","val_en_sentiment.csv","test_en_sentiment.csv"]:
        p = os.path.join(DATA_DIR, fn)
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        print("[ERROR] Missing files:", *missing, sep="\n - ")
        sys.exit(1)

def main():
    print(">>> Script started...")
    safe_check_files()
    set_seed(SEED)

    # แสดงตัวอย่าง 3 บรรทัดแรกของแต่ละ split เพื่อยืนยันโครงสร้าง
    for split in ["train_en_sentiment.csv","val_en_sentiment.csv","test_en_sentiment.csv"]:
        p = os.path.join(DATA_DIR, split)
        df_head = pd.read_csv(p).head(3)
        print(f"\n[{split}] preview:")
        print(df_head)

    # โหลด dataset จาก CSV
    data_files = {
        "train": f"{DATA_DIR}/train_en_sentiment.csv",
        "validation": f"{DATA_DIR}/val_en_sentiment.csv",
        "test": f"{DATA_DIR}/test_en_sentiment.csv"
    }
    raw = load_dataset("csv", data_files=data_files)

    # ตรวจว่าคอลัมน์ต้องมีจริง
    for name, ds in raw.items():
        cols = set(ds.column_names)
        if TEXT_COL not in cols or LABEL_COL not in cols:
            raise ValueError(f"[{name}] columns must contain '{TEXT_COL}' and '{LABEL_COL}', got {cols}")

    # แปลง label -> int และย้ายไปคีย์ 'labels' ที่ Trainer ใช้
    def to_labels(ex):
        # รองรับ label ที่เป็นสตริง/ตัวเลข
        ex["labels"] = int(ex[LABEL_COL])
        return ex

    raw = raw.map(to_labels)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    def tokenize_batch(examples):
        return tokenizer(examples[TEXT_COL], truncation=True, max_length=MAX_LEN)

    # tokenize และลบคอลัมน์ต้นฉบับ text/label ออก (กันชนกับคีย์ 'labels')
    encoded = raw.map(
        tokenize_batch,
        batched=True,
        remove_columns=[c for c in raw["train"].column_names if c in (TEXT_COL, LABEL_COL)]
    )

    # โมเดล
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label={0: "negative", 1: "neutral", 2: "positive"},
        label2id={"negative": 0, "neutral": 1, "positive": 2},
    )

    # Data collator (padding แบบไดนามิก)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Metrics
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
            "f1_weighted": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
        }

    use_fp16 = bool(torch.cuda.is_available())  # เปิดเฉพาะเมื่อมี GPU
    if use_fp16:
        print(">>> CUDA available. Using fp16.")
    else:
        print(">>> CUDA not available. Running on CPU (fp16 disabled).")

    # Training arguments (ใส่ logging ให้เห็นความคืบหน้าแน่นอน)
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=10,
        logging_first_step=True,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        save_total_limit=2,
        report_to="tensorboard",   # เปิด tensorboard (ดูได้ชัด)
        fp16=use_fp16,
        seed=SEED,
        dataloader_num_workers=2,
    )

    # สรุปจำนวนตัวอย่างแต่ละ split
    for name in ["train","validation","test"]:
        print(f">>> {name} size:", len(encoded[name]))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("\n***** Running training *****")
    trainer.train()
    print("***** Training finished *****\n")

    print("***** Evaluating on TEST *****")
    test_metrics = trainer.evaluate(encoded["test"])
    print(test_metrics)

    print("\n***** Saving model *****")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"All done. Model saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
