# Steam Reviews NLP — Sentiment Analysis & Topic Modeling

End-to-end NLP project on ~435,000 Steam game reviews (48 titles): exploratory data analysis, text preprocessing, classical and embedding-based topic modeling, and sentiment classification from a TF-IDF baseline up to a fine-tuned DistilBERT — with an explicit comparison of what the transformer buys and what it costs.

## Project Goals

1. **Sentiment classification** — predict whether a review recommends the game (Steam's "Recommended" / "Not Recommended" label).
2. **Topic modeling** — find cross-game themes (bugs, price, cheaters, community) rather than per-game clusters.
3. **Classical vs. transformer** — compare TF-IDF pipelines against DistilBERT on the same held-out test set and weigh score against cost.

## Pipeline / Notebooks

| Notebook | Content |
|---|---|
| `01_data_exploration` | EDA: class balance (~69% positive), corrupted `funny` values (uint32 wrap-around), language distribution (~81% English), vocabulary/Zipf analysis, review length vs. label, leakage diagnosis. Ends with a documented preprocessing decision table. |
| `02_preprocessing` | Shared cleaning (HTML unescape, lowercase, whitespace), language detection with `lingua` (keep EN + UNKNOWN), duplicate flagging, export of the cleaned corpus to Parquet. |
| `03_modelling` | Classical topic modeling: lemmatization (spaCy), stopword/leakage handling, coherence sweep over k, **LDA (k=10, counts)** vs. **NMF (k=15, TF-IDF)**. LDA k=10 chosen as the main model (9/10 clean, nameable topics). |
| `04_sentiment` | Sentiment with a *conservative* text variant (negations kept, bigrams, custom `token_pattern` for contractions). Group-aware stratified split (duplicates never straddle train/test). Models: **Logistic Regression (winner)**, LinearSVC, XGBoost ± metadata. Coefficient and error analysis, topic × sentiment join. |
| `05_transformer_sentiment` | DistilBERT two ways: frozen [CLS] embeddings + LogReg, and full fine-tuning (run on Colab T4; artifacts loaded locally). Evaluation on the identical test split, disagreement analysis, cost/benefit discussion. |
| `06_BERTopic` | Embedding-based topic modeling on a 25k sample: Sentence-BERT (`all-MiniLM-L6-v2`) → UMAP → HDBSCAN → c-TF-IDF. 44 auto-discovered topics + outlier bin, compared against LDA. |

## Key Results

Sentiment, evaluated on the same held-out test set (81,356 reviews, group-aware stratified 80/20 split):

| Model | macro-F1 | PR-AUC (negative class) |
|---|---|---|
| **TF-IDF + Logistic Regression** | **0.870** | **0.894** |
| LinearSVC | 0.868 | 0.891 |
| XGBoost (TF-IDF) | 0.807 | 0.877 |
| XGBoost (TF-IDF + metadata) | 0.814 | 0.883 |
| Frozen DistilBERT + LogReg | 0.817 | 0.824 |
| Fine-tuned DistilBERT (25k train subsample) | 0.868 | 0.894 |

> ⚠️ The two XGBoost rows predate a bug fix: `scale_pos_weight` was inverted (`n_pos/n_neg` instead of `n_neg/n_pos`), which up-weighted the majority class and likely amplified the negative-class recall collapse. The code in notebook 04 is fixed and the stale model caches were removed — re-running the notebook will refresh these numbers.

**Takeaways**

- The fine-tuned transformer **ties** the TF-IDF baseline on score — but reaches it with 1/8 of the training data, no feature engineering, and better handling of context (contrastive reviews, the "modding" vocabulary leakage). Its cost: GPU training, 268 MB weights, CPU inference infeasible. For a CPU-bound pipeline, the baseline remains the model to ship.
- Topic × sentiment join: overall negative share is 33.6%, but it ranges from **10.9%** (core gameplay/fun) to **68.7%** (modding-ban controversy) by topic — dissatisfaction clusters around nameable, addressable causes.
- Known caveat: part of the error rate is label noise ("best game 10/10" labeled Not Recommended), and some model weight sits on game-identity terms rather than pure sentiment (documented in notebooks 03–05).

## Repository Structure

```
├── data/                  # not in Git (see .gitignore)
│   ├── raw_data/          # steam_reviews.csv (source data)
│   ├── processed_data/    # cleaned corpus, lemma cache, coherence cache
│   ├── colab_data/        # embeddings & predictions computed on Colab
│   └── bert_topic_data/   # cached sentence embeddings
├── models/                # not in Git (see .gitignore)
│   ├── sentiment/         # TF-IDF vectorizer + LogReg / XGBoost models
│   ├── topic_models/      # LDA k=10, NMF k=15, vectorizers, doc-topic matrix
│   └── transformer_sentiment/  # fine-tuned DistilBERT checkpoint
├── notebooks/             # 01–06, ordered pipeline (see table above)
├── reports/               # pyLDAvis interactive topic visualization
├── src/
│   └── preprocessing.py   # shared, model-agnostic text cleaning
└── pyproject.toml         # dependencies (managed with uv)
```

## Setup

Requires Python ≥ 3.12. Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python -m spacy download en_core_web_sm
```

Place the raw dataset at `data/raw_data/steam_reviews.csv`, then run the notebooks in order (`01` → `06`). Expensive steps (lemmatization, coherence sweep, embeddings, fine-tuning) are cached to disk; the two GPU-dependent stages in notebook 05 were executed on Google Colab (T4) and their artifacts are loaded from `data/colab_data/`.

## Design Decisions Worth Noting

- **Two preprocessing variants** instead of one: an *aggressive* variant for topic modeling (stopwords, lemmatization, dedup, min length) and a *conservative* variant for sentiment (negations and short reviews kept, bigrams). The requirements conflict, so they are configured separately at the vectorizer level.
- **Group-aware splitting**: duplicate review texts share a `group_id`, and `StratifiedGroupKFold` keeps every group on one side of the split — no train/test leakage through repeated texts.
- **Leakage is diagnosed, then treated deliberately**: game-name tokens are stopworded; vocabulary-level leakage (e.g. Rocket League's "soccer") is documented with evidence instead of silently filtered.
- **`helpful` votes are excluded** as a feature: they accrue after posting and would leak post-hoc information.
