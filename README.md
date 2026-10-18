# Prompt Injection Detector

[![CI](https://github.com/ziko-asm/prompt-injection-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/ziko-asm/prompt-injection-detector/actions/workflows/ci.yml)

**Live results page:** https://prompt-injection-detector-snowy.vercel.app

How well can cheap, local detectors spot prompt injection in text, and do they still work on data from a different source?

This project compares detectors of increasing cost on public, labelled datasets of benign and prompt-injection text:

| Detector | Idea | Status |
|---|---|---|
| Keyword rule | n-grams learned automatically from training data (no hand-written phrases) | done |
| TF-IDF + logistic regression | word and character n-grams | done |
| Local LLM as judge | Llama 3.2 3B / Llama 3.1 8B via Ollama, zero-shot: asked to classify each text, never to follow it | done |
| Fine-tuned small transformer | multilingual DistilBERT fine-tuned on the GPU of a laptop (RTX 4050, 6 GB) | done |
| Cascade | transformer first; only uncertain texts go to Llama 3.1 8B | done |

Trained detectors are tested **in-domain** (train and test on the same dataset) and **cross-dataset** (train on one source, test on the other), which is where simple detectors usually break. LLM judges are not trained, so they are simply tested on both test sets (**zero-shot**).

All attack examples come from datasets already published by researchers on the Hugging Face Hub. This repository does not create attack text.

## Data
| Name | Source |
|---|---|
| `deepset` | [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) |
| `jackhhao` | [jackhhao/jailbreak-classification](https://huggingface.co/datasets/jackhhao/jailbreak-classification) |

Duplicates and empty texts are removed; train and test never share a text.

## Run
```bash
pip install -r requirements.txt
pytest -q                      # pipeline tests on a synthetic corpus
python -m src.data             # download and print dataset statistics
python -m src.evaluate         # benchmark -> results/results.json

# local LLM judges (needs Ollama running and the models pulled)
python -m src.llm_judge --check
python -m src.evaluate --detectors ollama:llama3.2:3b ollama:llama3.1:8b

# fine-tuned transformer (install the CUDA build of PyTorch first)
pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m src.evaluate --detectors transformer cascade
```
LLM rows are added to the existing results; answers are cached in `cache/`, so a re-run is instant.

## Results
F1 at threshold 0.5. For the LLM judges (never trained) the first two columns are simply their score on each test set. Full metrics (precision, recall, false positive rate, ROC-AUC) are in [`results/results.json`](results/results.json) and on the [project page](https://prompt-injection-detector-snowy.vercel.app).

| Detector | deepset → deepset | jackhhao → jackhhao | deepset → jackhhao | jackhhao → deepset | both → deepset | both → jackhhao | time / text |
|---|---|---|---|---|---|---|---|
| Keyword rule | 0.768 | 0.955 | 0.790 | 0.282 | 0.282 | 0.955 | 0.1 ms |
| TF-IDF + LR | 0.870 | 0.981 | 0.700 | 0.125 | 0.737 | 0.985 | 0.5 ms |
| Transformer (DistilBERT) | 0.957 | 0.978 | 0.696 | 0.373 | 0.857 | 0.981 | 1.5 ms |
| Cascade (transformer → 8B) | 0.947 | 0.985 | 0.701 | 0.442 | 0.846 | 0.977 | 192 ms |
| Llama 3.2 3B (zero-shot) | 0.286 | 0.791 | – | – | – | – | 2.4 s |
| Llama 3.1 8B (zero-shot) | 0.588 | 0.947 | – | – | – | – | 3.1 s |

**What this shows**

1. **Every trained detector overfits its source.** The fine-tuned transformer reaches 0.957 and 0.978 F1 in-domain, but trained on deepset and tested on jackhhao it blocks 97% of harmless texts, and in the other direction it finds only 23% of injections. TF-IDF and the keyword rule fail the same way.
2. **Training on both sources closes most of the gap** (transformer: 0.857 and 0.981), at a small cost on deepset. Two sources are still only two sources; a third, unseen one is the honest next test.
3. **Only the LLM judge transfers without training.** Llama 3.1 8B scores 0.947 on jackhhao with a 2.5% false positive rate, but finds only 42% of deepset's short, partly German injections, and it is about 2,000 times slower than the transformer on a laptop GPU. Llama 3.2 3B is much weaker.
4. **A cascade cannot rescue a confidently wrong model.** Sending only the transformer's uncertain texts (score between 0.05 and 0.95) to the 8B judge costs 1–6% LLM calls and keeps in-domain accuracy. Cross-dataset, the transformer is confident and wrong, so few texts reach the LLM and the errors remain. Confidence-based routing does not detect distribution shift.

**Limitations:** small test sets (116 and 257 texts), one seed per model, two sources, and the LLM judge itself was not attacked.

By [Ziya Azizov](https://www.linkedin.com/in/ziya-azizov-91bba638a).
