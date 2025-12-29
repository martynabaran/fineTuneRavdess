# -*- coding: utf-8 -*-
"""
fineTuningRavdess.py

Przerobiony skrypt z notebooka Colab
Do uruchomienia na PLGrid
Datasety i wyniki przechowywane w $SCRATCH
Autor: Martyna Baran, GGNS 2025
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import glob
import shutil

from datasets import load_dataset, Dataset, Audio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score
)
from transformers import (
    Wav2Vec2Processor,
    TrainerCallback,
    Wav2Vec2ForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

# =========================================================
# 0. PLGRID SCRATCH PATHS
# =========================================================
import os

# =========================================================
# 0. PLGRID SCRATCH PATHS
# =========================================================
# Jeśli jest MEMFS (RAM disk), użyj go; inaczej standardowy SCRATCH
SCRATCH = os.environ.get("MEMFS", os.environ.get("SCRATCH", "/tmp"))
HF_CACHE = os.path.join(SCRATCH, "huggingface_cache")
DATASET_DIR = os.path.join(SCRATCH, "ravdess")
DATASET_DIR2 = os.path.join("/net/tscratch/people/plgmarbar/", "ravdess")
# Ścieżka do folderu z danymi (rozpakowany ZIP)
RAVDESS_DIR = os.path.join("/net/tscratch/people/plgmarbar/ravdess", "ravdess_audio_only")
CSV_PATH = os.path.join(RAVDESS_DIR, "metadata_with_emotions.csv")
OUTPUT_DIR = os.path.join("/net/tscratch/people/plgmarbar/ravdess", "wav2vec2_checkpoints")
OUTPUT_DIR_PERSISTENT=OUTPUT_DIR
roc_dir = os.path.join(OUTPUT_DIR, "roc")

os.makedirs(roc_dir, exist_ok=True)
os.makedirs(OUTPUT_DIR_PERSISTENT, exist_ok=True)
os.makedirs(HF_CACHE, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set Hugging Face cache
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

print("[INFO] Scratch directories configured:")
print(f"  SCRATCH:       {SCRATCH}")
print(f"  DATASET_DIR:   {DATASET_DIR}")
print(f"  HF_CACHE:      {HF_CACHE}")
print(f"  OUTPUT_DIR:    {OUTPUT_DIR}")

# =========================================================
# 1. Metric Computation
# =========================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    bal_acc = balanced_accuracy_score(labels, preds)
    return {"balanced_accuracy": bal_acc}

# =========================================================
# 2. Data Collator
# =========================================================
class DataCollatorWithPadding:
    def __init__(self, processor, padding=True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"], "attention_mask": f["attention_mask"]} for f in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.long)
        return batch

class BestEpochRocCollector(TrainerCallback):
    def __init__(self, trainer, dataset_dict, save_dir):
        self.trainer = trainer
        self.dataset_dict = dataset_dict
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_bal_acc = -1
        self.best_epoch_data = None

        print(f"[ROC CALLBACK INIT] Save dir={self.save_dir}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch = state.epoch
        print(f"[CALLBACK] on_evaluate triggered | epoch={epoch}")

        # Jeśli trainer nie jest jeszcze przypisany
        if self.trainer is None:
            print("[WARN] trainer reference is None! Callback won’t work.")
        
        if metrics is None:
            print("[WARN] metrics=None (Trainer did not return any metrics)")
            return control

        print(f"[CALLBACK] metrics received: {metrics}")
        bal_acc = metrics.get("eval_balanced_accuracy", None)

        #bal_acc = metrics.get("balanced_accuracy", None)
        if bal_acc is None:
            print("[CALLBACK] ⚠ balanced_accuracy NOT FOUND in metrics.")
            print(f"Available keys: {list(metrics.keys())}")
            return control

        print(f"[CALLBACK] Balanced accuracy for epoch {epoch}: {bal_acc:.4f}")

        # Jeśli ta epoka jest najlepsza → zapisujemy predykcje
        if bal_acc > self.best_bal_acc:
            print(
                f"[CALLBACK]  NEW BEST epoch found! "
                f"prev={self.best_bal_acc:.4f} -> new={bal_acc:.4f}"
            )

            self.best_bal_acc = bal_acc

            print("[CALLBACK] Running trainer.predict() on validation dataset...")
            preds = self.trainer.predict(self.dataset_dict["validation"])
            logits = preds.predictions
            labels = preds.label_ids

            print("[CALLBACK] Predictions collected. Saving in-memory...")
            self.best_epoch_data = {
                "epoch": int(epoch),
                "labels": labels,
                "logits": logits,
                "balanced_accuracy": float(bal_acc),
            }
        else:
            print(
                f"[CALLBACK] No improvement. Best remains {self.best_bal_acc:.4f}"
            )

        return control

    def save_best(self):
        print("[CALLBACK] save_best() called.")
    
        if self.best_epoch_data is None:
            print("[CALLBACK]  No best_epoch_data → nothing to save.")
            return
    
        # --- SAFE JSON CONVERSION ---
        # Konwersja wszystkich numpy → list
        safe_data = {}
        for k, v in self.best_epoch_data.items():
            if hasattr(v, "tolist"):   # numpy, tensors itp.
                safe_data[k] = v.tolist()
            else:
                safe_data[k] = v
    
        # --- SAVE NUMPY FILES ---
        print("[CALLBACK] Saving NUMPY files for best epoch...")
    
        # Save logits/labels only if present
        if "logits" in self.best_epoch_data:
            np.save(os.path.join(self.save_dir, "best_logits.npy"), self.best_epoch_data["logits"])
        if "labels" in self.best_epoch_data:
            np.save(os.path.join(self.save_dir, "best_labels.npy"), self.best_epoch_data["labels"])
    
        # --- SAVE JSON ---
        with open(os.path.join(self.save_dir, "best_epoch.json"), "w") as f:
            json.dump(safe_data, f, indent=2)
    
        # --- PRINT SUMMARY ---
        epoch = safe_data.get("epoch", "-")
        bal = safe_data.get("balanced_accuracy", None)
        bal_msg = f"{bal:.4f}" if isinstance(bal, (int, float)) else str(bal)
    
        print(
            f"[CALLBACK] BEST EPOCH SAVED:\n"
            f"    epoch = {epoch}\n"
            f"    bal_acc = {bal_msg}\n"
            f"    path = {self.save_dir}"
        )




# =========================================================
# 3. Dataset preparation
# =========================================================

print(f"[INFO] Using local dataset from: {RAVDESS_DIR}")
print(f"[INFO] Loading metadata from: {CSV_PATH}")

# Wczytaj DataFrame
df = pd.read_csv(CSV_PATH)
print(f"[INFO] Loaded {len(df)} samples")

# Mapowanie etykiet na ID
unique_labels = sorted(df["label"].unique())
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
df["label_id"] = df["label"].map(label2id)

print("[INFO] Label mapping:")
for k, v in label2id.items():
    print(f"  {k} -> {v}")

# Podział danych
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label_id"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label_id"], random_state=42)

print(f"[INFO] Train/Val/Test sizes: {len(train_df)} / {len(val_df)} / {len(test_df)}")

# Konwersja Pandas → HuggingFace Dataset
def df_to_hf_dataset(df, base_dir):

    df = df.copy()
    df["full_path"] = df["path"].apply(lambda x: os.path.join(base_dir, x))

    ds = Dataset.from_pandas(df[["full_path", "label_id"]])
    ds = ds.rename_column("full_path", "audio")
    ds = ds.rename_column("label_id", "label")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds
    
RAVDESS_DIR2 = "/net/tscratch/people/plgmarbar/ravdess"

train_ds = df_to_hf_dataset(train_df, RAVDESS_DIR2)
val_ds = df_to_hf_dataset(val_df, RAVDESS_DIR2)
test_ds = df_to_hf_dataset(test_df, RAVDESS_DIR2)

print("[INFO] HuggingFace Datasets created successfully:")
print(f"  Train: {len(train_ds)} samples")
print(f"  Val:   {len(val_ds)} samples")
print(f"  Test:  {len(test_ds)} samples")

# =========================================================
# 4. Processor & preprocessing
# =========================================================

print("[INFO] Loading Wav2Vec2 processor...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
processor.feature_extractor.return_attention_mask = True
print("[INFO] Processor loaded successfully.")


def preprocess_batch(batch):
    audio_arrays = [a["array"] for a in batch["audio"]]
    sampling_rate = batch["audio"][0]["sampling_rate"]
    inputs = processor(
        audio_arrays,
        sampling_rate=sampling_rate,
        return_tensors="np",
        padding=True,
        return_attention_mask=True
    )
    batch["input_values"] = inputs["input_values"]
    batch["attention_mask"] = inputs["attention_mask"]
    return batch
print("/n Start processing")
train_ds = train_ds.map(preprocess_batch, batched=True, batch_size=4, remove_columns=["audio"])
val_ds = val_ds.map(preprocess_batch, batched=True, batch_size=4, remove_columns=["audio"])
test_ds = test_ds.map(preprocess_batch, batched=True, batch_size=4, remove_columns=["audio"])
print("Finished processing")
data_collator = DataCollatorWithPadding(processor=processor, padding=True)


dataset_dict = {
    "train": train_ds,
    "validation": val_ds,
    "test": test_ds
}

# =========================================================
# 5. Model initialization
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Initializing Wav2Vec2 model...")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label,
    attention_dropout=0.1,
    hidden_dropout=0.1,
    mask_time_prob=0.05,
    gradient_checkpointing=True
)
model.freeze_feature_extractor()
model.to(device)
print("[INFO] Model initialized.")


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    load_best_model_at_end=True,
    metric_for_best_model="balanced_accuracy",
    greater_is_better=True,
    logging_dir="./logs",
    logging_strategy="epoch",
    report_to="none",
    seed=42,
    fp16=torch.cuda.is_available(),
)


print(f"[INFO] Using device: {device}")

# =========================================================
# 6. Training setup
# =========================================================

roc_callback = BestEpochRocCollector(trainer=None, dataset_dict=dataset_dict, save_dir=roc_dir)

print("[INFO] Setting up Trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5), roc_callback],
)
roc_callback.trainer = trainer  # <-- dopinamy callback teraz

print("[INFO] Trainer is ready.")

# =========================================================
# 7. Training
# =========================================================
print("[INFO] Starting training...")
trainer.train()
print("[INFO] Training finished.")

# =========================================================
# 8. Evaluation – SKIPPED (oszczędzanie zasobów)
# =========================================================
print("[INFO] Skipping full evaluation metrics to save time/resources.")

print("[INFO] Saving BEST EPOCH prediction data...")
roc_callback.save_best()

# ===== ROC CURVE PLOT =====
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

best_logits = np.load(os.path.join(OUTPUT_DIR, "roc", "best_logits.npy"))
best_labels = np.load(os.path.join(OUTPUT_DIR, "roc", "best_labels.npy"))

# prawdopodobieństwa
probs = torch.softmax(torch.tensor(best_logits), dim=1).numpy()

# Jeśli multi-class
if probs.shape[1] > 2:
    fpr, tpr, roc_auc = {}, {}, {}
    plt.figure()
    for cls in range(probs.shape[1]):
        fpr[cls], tpr[cls], _ = roc_curve(best_labels == cls, probs[:, cls])
        roc_auc[cls] = auc(fpr[cls], tpr[cls])
        emotion_name = id2label[cls]  # id2label z Twojego wcześniejszego mappingu
        plt.plot(fpr[cls], tpr[cls], label=f"{emotion_name} (AUC={roc_auc[cls]:.2f})")
        # plt.plot(fpr[cls], tpr[cls], label=f"class {cls} AUC={roc_auc[cls]:.2f}")

else:  # binary
    fpr, tpr, _ = roc_curve(best_labels, probs[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC={roc_auc:.2f})")

plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Best Epoch")
plt.legend()

roc_path = os.path.join(OUTPUT_DIR, "roc", "roc_best_epoch.png")
plt.savefig(roc_path)
plt.close()
print(f"[INFO] ROC curve saved: {roc_path}")



