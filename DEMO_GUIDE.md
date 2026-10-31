# CineMatch live demonstration guide

This file is a suggested run order and explanation outline for the group demonstration. Adjust timing and the division of speaking roles to match the actual work completed by each member.

## 1. Before the demonstration

Use Python 3.10 or newer. Open a terminal in the `movie_recommender_nlp` directory and run:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Run the offline demo

The project already includes a synthetic demo dataset. To regenerate it and demonstrate the command-line recommender:

```bash
python scripts/generate_demo_data.py
python run_demo.py
```

To regenerate the experiment metrics and graphs:

```bash
python run_evaluation.py
```

To launch the interactive UI:

```bash
streamlit run app.py
```

The app opens locally in a browser. Select a user, choose `hybrid`, `content`, `bpr`, `intent`, or `popularity`, set the number of recommendations, and show the movie explanations. Open the rating-history expander to explain how past ratings shape a profile.

## 3. Recommended explanation sequence

1. **Problem:** users face a large movie catalog; ratings alone can make a recommendation hard to explain.
2. **Research connection:** PRR-LI motivates text-plus-dimension enrichment; PETER motivates user-item-personalized explanations; ANTM motivates latent intents, attention, and BPR ranking.
3. **Pre-processing:** normalize movie text, map genre and lexical cues to dimensions, then concatenate the text and dimension labels.
4. **Models:** TF-IDF computes content relevance; NMF discovers latent topics; a softmax gives user-specific attention over topics; BPR learns positive-versus-negative preferences; the weighted hybrid score combines them.
5. **Output:** show ranked unseen movies and read one explanation aloud. Explain that it is generated from the user's positive-history dimensions and the movie's dimensions.
6. **Evaluation:** open `results/evaluation_metrics.csv` and the three PNG graphs. Explain HR@K, NDCG@K, MAP@10, and Precision@10. With one held-out relevant movie per user, HR@K and Recall@K are equal in this experiment.
7. **Limitations:** the included ratings are synthetic; the implementation is a compact adaptation, not a reproduction of the papers' full neural architectures; unobserved movies are treated as non-relevant in the ranking calculation.

## 4. Current included evaluation result

The fixed-seed demo run evaluates 90 users, holding out one positive movie per user:

| Method | HR@5 | HR@10 | NDCG@5 | NDCG@10 | MAP@10 | Precision@10 |
|---|---:|---:|---:|---:|---:|---:|
| Popularity | 0.233 | 0.556 | 0.122 | 0.224 | 0.126 | 0.056 |
| Content-only | 0.411 | 0.744 | 0.285 | 0.391 | 0.287 | 0.074 |
| BPR-only | 0.267 | 0.567 | 0.151 | 0.246 | 0.151 | 0.057 |
| Hybrid | 0.444 | 0.700 | 0.291 | 0.372 | 0.274 | 0.070 |

The hybrid is best on HR@5 and NDCG@5 in this run, while content-only is strongest at K=10. Do not claim that the hybrid wins on every metric.

## 5. Reproducibility and testing

```bash
python -m unittest discover -s tests -v
```

The packaged run passed all four included unit tests in the build environment. Run the tests again after making changes.

## 6. Dataset warning

The default dataset has 50 movies, 90 synthetic users, and 3,060 generated ratings. It is intentionally self-contained for an offline demonstration. It is not a public benchmark and is not evidence of real-world performance. An optional MovieLens 100K downloader is included:

```bash
python scripts/download_movielens.py
```

For a real-data experiment, set `MOVIE_DATA_DIR=data/ml-100k` before running the evaluation and report that run separately. MovieLens 100K does not provide plot descriptions in its item file, so the downloader generates short genre-derived text; richer movie text would improve the NLP features.

## 7. Final submission checks

- Confirm the group number on the report cover.
- Fill in each member's actual contribution and collect any required signatures.
- Ensure all members can explain and run the code themselves.
- Verify the AI tool declaration and update it if the final use of tools changes.
- Follow the faculty's instructions for final submission and presentation.
