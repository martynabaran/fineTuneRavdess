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
    Wav2Vec2ForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)


# =========================================================
# 0. PLGRID SCRATCH PATHS
# =========================================================
SCRATCH = os.environ.get("SCRATCH", "/tmp")
HF_CACHE = os.path.join(SCRATCH, "huggingface_cache")
DATASET_DIR = os.path.join(SCRATCH, "ravdess_dataset")
OUTPUT_DIR = os.path.join(SCRATCH, "wav2vec2_checkpoints")

os.makedirs(HF_CACHE, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set Hugging Face cache
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
# =========================================================
# 1. Metric Computation
# =========================================================
def compute_ser_metrics(y_true, y_pred, y_score, labels=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = list(np.unique(np.concatenate([y_true, y_pred])))
    labels = list(labels)
    n_classes = len(labels)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    y_true_idx = np.array([label_to_idx[l] for l in y_true])
    y_pred_idx = np.array([label_to_idx[l] for l in y_pred])

    if y_score is None:
        y_score = np.zeros((len(y_pred_idx), n_classes))
        y_score[np.arange(len(y_pred_idx)), y_pred_idx] = 1.0
    else:
        y_score = np.asarray(y_score)

    acc = float(accuracy_score(y_true_idx, y_pred_idx))
    bal_acc = float(balanced_accuracy_score(y_true_idx, y_pred_idx))
    p_r_f_support = precision_recall_fscore_support(y_true_idx, y_pred_idx, labels=range(n_classes), zero_division=0)
    precision_per, recall_per, f1_per, support_per = p_r_f_support

    precision_macro = float(np.mean(precision_per))
    recall_macro = float(np.mean(recall_per))
    f1_macro = float(np.mean(f1_per))

    p_r_f_micro = precision_recall_fscore_support(y_true_idx, y_pred_idx, average='micro', zero_division=0)
    precision_micro, recall_micro, f1_micro = map(float, p_r_f_micro[:3])

    p_r_f_weighted = precision_recall_fscore_support(y_true_idx, y_pred_idx, average='weighted', zero_division=0)
    precision_weighted, recall_weighted, f1_weighted = map(float, p_r_f_weighted[:3])

    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=range(n_classes))
    specificity_per = []
    per_class = {}
    for i, lab in enumerate(labels):
        TP = int(cm[i, i])
        FN = int(cm[i, :].sum() - TP)
        FP = int(cm[:, i].sum() - TP)
        TN = int(cm.sum() - (TP + FP + FN))
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        specificity_per.append(specificity)
        per_class[lab] = {
            "precision": float(precision_per[i]),
            "recall": float(recall_per[i]),
            "f1": float(f1_per[i]),
            "support": int(support_per[i]),
            "specificity": float(specificity),
            "AP": None,
            "AUC": None,
            "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        }

    try:
        mcc = float(matthews_corrcoef(y_true_idx, y_pred_idx))
    except Exception:
        mcc = None

    y_true_bin = label_binarize(y_true_idx, classes=list(range(n_classes)))
    aucs, aps = [], []
    for i in range(n_classes):
        y_true_i, y_score_i = y_true_bin[:, i], y_score[:, i]
        try:
            auc_i = roc_auc_score(y_true_i, y_score_i)
        except Exception:
            auc_i = None
        try:
            ap_i = average_precision_score(y_true_i, y_score_i)
        except Exception:
            ap_i = None
        per_class[labels[i]]["AUC"] = auc_i
        per_class[labels[i]]["AP"] = ap_i
        if auc_i is not None:
            aucs.append(auc_i)
        if ap_i is not None:
            aps.append(ap_i)

    auc_macro = float(np.mean(aucs)) if aucs else None
    mAP_macro = float(np.mean(aps)) if aps else None

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "mcc": mcc,
        "auc_macro": auc_macro,
        "mAP_macro": mAP_macro,
        "per_class": per_class,
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
    }
    return metrics

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

# =========================================================
# 3. Dataset preparation
# =========================================================
dataset = load_dataset("narad/ravdess", cache_dir=DATASET_DIR)

# Convert dataset to train/val/test splits
df = pd.DataFrame(dataset['train'])  # narad/ravdess ma tylko split "train" w HF
if "labels" in df.columns:
    df = df.rename(columns={"labels": "label"})

unique_labels = sorted(df["label"].unique())
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
df["label_id"] = df["label"].map(label2id)

train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label_id"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label_id"], random_state=42)

def df_to_hf_dataset(df):
    ds = Dataset.from_pandas(df[["path", "label_id"]])
    ds = ds.rename_column("path", "audio")
    ds = ds.rename_column("label_id", "label")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds

train_ds = df_to_hf_dataset(train_df)
val_ds = df_to_hf_dataset(val_df)
test_ds = df_to_hf_dataset(test_df)

# =========================================================
# 4. Processor & preprocessing
# =========================================================
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
processor.feature_extractor.return_attention_mask = True

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

train_ds = train_ds.map(preprocess_batch, batched=True, batch_size=2, remove_columns=["audio"])
val_ds = val_ds.map(preprocess_batch, batched=True, batch_size=2, remove_columns=["audio"])
test_ds = test_ds.map(preprocess_batch, batched=True, batch_size=2, remove_columns=["audio"])

data_collator = DataCollatorWithPadding(processor=processor, padding=True)

dataset_dict = {
    "train": train_ds,
    "validation": val_ds,
    "test": test_ds
}

# =========================================================
# 5. Model initialization
# =========================================================
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label,
    attention_dropout=0.1,
    hidden_dropout=0.1,
    mask_time_prob=0.05
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# =========================================================
# 6. Training setup
# =========================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    learning_rate=1e-4,
    lr_scheduler_type="reduce_lr_on_plateau",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=50,
    load_best_model_at_end=True,
    metric_for_best_model="balanced_accuracy",
    greater_is_better=True,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_strategy="epoch",
    report_to="none",
    seed=42,
    fp16=torch.cuda.is_available(),
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    y_pred = np.argmax(logits, axis=1)
    metrics = compute_ser_metrics(
        y_true=labels,
        y_pred=y_pred,
        y_score=logits,
        labels=list(range(len(label2id))),
    )
    return {
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "f1_macro": metrics["f1_macro"],
        "mcc": metrics["mcc"],
        "auc_macro": metrics["auc_macro"],
        "mAP_macro": metrics["mAP_macro"],
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

# =========================================================
# 7. Training
# =========================================================
trainer.train()

# =========================================================
# 8. Evaluation & saving metrics
# =========================================================
def metrics_to_dataframe(metrics):
    rows = [{
        "label": "GLOBAL",
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "f1_macro": metrics["f1_macro"],
        "mcc": metrics["mcc"],
        "auc_macro": metrics["auc_macro"],
        "mAP_macro": metrics["mAP_macro"],
    }]
    for lab, per in metrics["per_class"].items():
        rows.append({
            "label": lab,
            "precision": per["precision"],
            "recall": per["recall"],
            "f1": per["f1"],
            "specificity": per["specificity"],
            "AP": per["AP"],
            "AUC": per["AUC"],
            "TP": per["TP"],
            "FP": per["FP"],
            "TN": per["TN"],
            "FN": per["FN"],
        })
    return pd.DataFrame(rows)

for split_name, split_ds in [("train", dataset_dict["train"]),
                             ("val", dataset_dict["validation"]),
                             ("test", dataset_dict["test"])]:
    preds = trainer.predict(split_ds)
    logits, labels = preds.predictions, preds.label_ids
    y_pred = np.argmax(logits, axis=1)

    metrics = compute_ser_metrics(
        y_true=labels,
        y_pred=y_pred,
        y_score=logits,
        labels=list(range(len(label2id))),
    )

    df_metrics = pd.DataFrame(metrics_to_dataframe(metrics))
    df_metrics.to_csv(
        os.path.join(OUTPUT_DIR, f"wav2vec2_{split_name}_metrics.csv"), index=False
    )

    with open(os.path.join(OUTPUT_DIR, f"wav2vec2_{split_name}_summary.json"), "w") as f:
        json.dump(metrics, f, indent=2)
