# -*- coding: utf-8 -*-
import os, sys
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
    TrainerCallback,
)
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# -----------------------
# CONFIG
# -----------------------
MODEL_NAME = "FacebookAI/xlm-roberta-base"
DATA_DIR = "data"   
TEXT_COL = "text"
LABEL_COL = "label"  # int: 0=neg,1=neu,2=pos
NUM_LABELS = 3
OUTPUT_DIR = "./xlmr-finetuned-course-sentiment"
MAX_LEN = 256
SEED = 42
# -----------------------

def check_files():
    missing = []
    for fn in ["train_en_sentiment.csv","val_en_sentiment.csv","test_en_sentiment.csv"]:
        p = os.path.join(DATA_DIR, fn)
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        print("[ERROR] Missing files:\n - " + "\n - ".join(missing))
        sys.exit(1)

def main():
    print(">>> Script started")
    set_seed(SEED)
    check_files()

    
    for fn in ["train_en_sentiment.csv","val_en_sentiment.csv","test_en_sentiment.csv"]:
        p = os.path.join(DATA_DIR, fn)
        print(f"\n[{fn}] preview:")
        print(pd.read_csv(p).head(3))

    # Load CSVs
    data_files = {
        "train": f"{DATA_DIR}/train_en_sentiment.csv",
        "validation": f"{DATA_DIR}/val_en_sentiment.csv",
        "test": f"{DATA_DIR}/test_en_sentiment.csv",
    }
    raw = load_dataset("csv", data_files=data_files)

    # map label -> 'labels' (int)
    def to_labels(ex):
        ex["labels"] = int(ex[LABEL_COL])
        return ex
    raw = raw.map(to_labels)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    def tok_fn(batch):
        return tokenizer(batch[TEXT_COL], truncation=True, max_length=MAX_LEN)

    cols_to_remove = [c for c in raw["train"].column_names if c in (TEXT_COL, LABEL_COL)]
    enc = raw.map(tok_fn, batched=True, remove_columns=cols_to_remove)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label={0:"negative",1:"neutral",2:"positive"},
        label2id={"negative":0,"neutral":1,"positive":2},
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    
    try:
        import evaluate
        acc_metric = evaluate.load("accuracy")
        f1_metric  = evaluate.load("f1")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
                "f1_macro": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
            }
    except Exception:
        from sklearn.metrics import accuracy_score, f1_score
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": float(accuracy_score(labels, preds)),
                "f1_macro": float(f1_score(labels, preds, average="macro")),
            }

    use_fp16 = bool(torch.cuda.is_available())
    if use_fp16:
        print(">>> CUDA available. Using fp16.")
    else:
        print(">>> CUDA not available. Running on CPU (fp16 disabled).")

   
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=500,
        fp16=use_fp16,
        seed=SEED,
        
    )
   

    print(f">>> sizes -> train={len(enc['train'])}, val={len(enc['validation'])}, test={len(enc['test'])}")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=enc["train"],
        eval_dataset=enc["validation"],  
        tokenizer=tokenizer,              
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    
    class EpochEndEval(TrainerCallback):
        def __init__(self, trainer_ref, enc_datasets):
            self.trainer = trainer_ref     
            self.enc = enc_datasets
            self.history = []

        def on_epoch_end(self, args, state, control, **kwargs):
            
            val = self.trainer.evaluate(self.enc["validation"])
            tst = self.trainer.evaluate(self.enc["test"])
            epoch_num = int(state.epoch) if state.epoch is not None else len(self.history) + 1

            row = {
                "epoch": epoch_num,
                "val_accuracy": val.get("eval_accuracy", float("nan")),
                "val_f1_macro": val.get("eval_f1_macro", float("nan")),
                "test_accuracy": tst.get("eval_accuracy", float("nan")),
                "test_f1_macro": tst.get("eval_f1_macro", float("nan")),
            }
            print(f"[Epoch {row['epoch']}] "
                  f"VAL acc={row['val_accuracy']:.4f} f1={row['val_f1_macro']:.4f} | "
                  f"TEST acc={row['test_accuracy']:.4f} f1={row['test_f1_macro']:.4f}")
            self.history.append(row)

    cb = EpochEndEval(trainer, enc)
    trainer.add_callback(cb)

   
    print("\n***** Running training *****")
    trainer.train()
    print("***** Training finished *****\n")

   
    if cb.history:
        df = pd.DataFrame(cb.history)
        df.to_csv("metrics_per_epoch.csv", index=False)
        print("Saved: metrics_per_epoch.csv")

        
        plt.figure(figsize=(8,5))
        plt.plot(df["epoch"], df["val_accuracy"], marker="o", label="Val Accuracy")
        plt.plot(df["epoch"], df["val_f1_macro"], marker="o", label="Val F1")
        plt.plot(df["epoch"], df["test_accuracy"], marker="s", label="Test Accuracy")
        plt.plot(df["epoch"], df["test_f1_macro"], marker="s", label="Test F1")
        plt.xlabel("Epoch"); plt.ylabel("Score"); plt.title("Accuracy & F1 per Epoch")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig("accuracy_f1_per_epoch.png", dpi=150)
        print("Saved: accuracy_f1_per_epoch.png")
       
        
    else:
        print("[WARN] callback history ว่าง (ไม่มีบันทึกต่อ epoch)")

    # --------- FINAL EVAL: VALIDATION / TEST ----------
    print("***** Evaluating on VALIDATION (final) *****")
    val_metrics = trainer.evaluate(enc["validation"])
    print(f"VAL -> acc={val_metrics['eval_accuracy']:.4f} | f1_macro={val_metrics['eval_f1_macro']:.4f}")

    print("***** Evaluating on TEST (final) *****")
    test_metrics = trainer.evaluate(enc["test"])
    print(f"TEST -> acc={test_metrics['eval_accuracy']:.4f} | f1_macro={test_metrics['eval_f1_macro']:.4f}")

    # --------- รายงานละเอียด (TEST) ----------
    preds = trainer.predict(enc["test"])
    y_true = np.array(enc["test"]["labels"])
    y_pred = np.argmax(preds.predictions, axis=-1)

    print("\nClassification report (TEST)")
    print(classification_report(
        y_true, y_pred,
        target_names=["negative","neutral","positive"],
        digits=4
    ))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))

   
    print("\n***** Saving model *****")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"All done. Model saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
