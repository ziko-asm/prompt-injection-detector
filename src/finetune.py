"""
Fine-tuned small transformer detector.

A pretrained multilingual encoder (DistilBERT, 134M parameters; one of the
datasets contains German text) gets a classification head and is fine-tuned on
the training split. It runs on a laptop GPU (RTX 4050, 6 GB) in a few minutes
and falls back to CPU, slowly, if no GPU is found.

    python -m src.evaluate --detectors transformer

Needs torch and transformers (see README for the CUDA install command).
"""

from __future__ import annotations

import random

import numpy as np

DEFAULT_MODEL = "distilbert-base-multilingual-cased"


class FineTunedTransformer:
    name = "transformer"

    def __init__(self, model_name: str = DEFAULT_MODEL, epochs: int = 4, lr: float = 3e-5,
                 batch_size: int = 16, max_len: int = 256, seed: int = 0, device: str | None = None):
        self.model_name, self.epochs, self.lr = model_name, epochs, lr
        self.batch_size, self.max_len, self.seed = batch_size, max_len, seed
        self.device = device

    def _setup(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.torch = torch
        self.device_ = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok_ = AutoTokenizer.from_pretrained(self.model_name)
        self.model_ = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2).to(self.device_)

    def _encode(self, texts):
        return self.tok_(list(texts), truncation=True, max_length=self.max_len,
                         padding=True, return_tensors="pt").to(self.device_)

    def fit(self, texts, labels):
        from transformers import get_linear_schedule_with_warmup

        self._setup()
        torch = self.torch
        texts, labels = list(texts), np.asarray(labels)
        # class weights, so an unbalanced training set does not bias the threshold
        counts = np.bincount(labels, minlength=2).astype(float)
        weights = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float,
                               device=self.device_)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=0.01)
        steps = self.epochs * int(np.ceil(len(texts) / self.batch_size))
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        use_amp = self.device_ == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        rng = np.random.default_rng(self.seed)

        self.model_.train()
        for epoch in range(self.epochs):
            order = rng.permutation(len(texts))
            total = 0.0
            for i in range(0, len(order), self.batch_size):
                idx = order[i:i + self.batch_size]
                batch = self._encode([texts[j] for j in idx])
                y = torch.as_tensor(labels[idx], device=self.device_)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    loss = loss_fn(self.model_(**batch).logits.float(), y)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                total += loss.item() * len(idx)
            print(f"  transformer epoch {epoch + 1}/{self.epochs}  loss {total / len(texts):.4f}"
                  f"  ({self.device_})", flush=True)
        return self

    def score(self, texts) -> np.ndarray:
        torch = self.torch
        texts = list(texts)
        self.model_.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                batch = self._encode(texts[i:i + 64])
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=self.device_ == "cuda"):
                    logits = self.model_(**batch).logits.float()
                out.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
        if self.device_ == "cuda":
            torch.cuda.synchronize()
        return np.concatenate(out) if out else np.empty(0)
