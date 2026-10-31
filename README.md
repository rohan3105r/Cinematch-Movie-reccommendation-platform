# CineMatch NLP: Explainable Personalized Movie Recommendation

**Mini-project theme:** NLP + personalized recommendation + explainable ranking  
**Research foundation:** PRR-LI (NAACL 2024), PETER (ACL-IJCNLP 2021), and ANTM (Knowledge-Based Systems, 2025).

**Group details from the uploaded Activity 1 report:** Adnan Shaikh (16030726008), Aaryan Raut (16030726018), and Rohan Rahalkar (16030726011). Confirm all member details and enter the group number before submission.

## 1. What this project does

CineMatch recommends unseen movies for a selected user. It reads movie descriptions and genres, enriches the text with interpretable dimensions (for example, story, acting, visuals, music, suspense, romance, and emotion), learns user preferences from ratings, ranks candidate movies, and produces a short explanation tied to the user and recommended movie.

The proposed ranking combines:

- **BPR collaborative ranking** learned from positive and negative ratings: 20%.
- **TF-IDF content similarity** between a user's positively rated movies and a candidate: 65%.
- **Attentive latent-intent similarity** from NMF topic distributions: 15%.

Each component is min-max normalized over the current candidate set before combination. The weights are transparent project hyperparameters, not values copied from the papers.

## 2. Research-paper connection and implementation boundary

| Supplied paper | Main idea used | What this project implements |
|---|---|---|
| Xu & Xu (2024), PRR-LI | Mine explicit/implicit review dimensions and encode `text + dimension` together. | A deterministic aspect lexicon and genre-to-dimension mapping enrich movie overview text; the combined representation is TF-IDF encoded. No LLM-based mining is used. |
| Li, Zhang & Chen (2021), PETER | User/item IDs contribute to personalized explanations through context prediction and joint recommendation/explanation learning. | A user-item-conditioned natural-language template uses the user's learned dimension interests and the candidate movie's dimensions. This is not a trained Transformer/PETER reproduction. |
| Zhang et al. (2025), ANTM | Discover user latent intents with a neural topic model, weight intents with attention, and use BPR for personalized ranking. | NMF provides an interpretable topic-model proxy; softmax attention weights topics from each user's positive history; a BPR matrix-factorization model learns pairwise ranking. NMF is not the paper's variational neural topic model. |

The implementation is deliberately a compact, explainable adaptation suitable for a student mini-project. It should not be described as a reproduction of the original architectures.

## 3. Setup

Python 3.10 or later is recommended.

### Windows: one-click startup

After downloading or extracting the project, open the `movie_recommender_nlp` folder and double-click `Start_CineMatch.bat`. On first use, it creates the project's `.venv`, installs missing dependencies, and recreates the offline demo data if needed. It then finds an available port between 8501 and 8510, starts Streamlit, and opens the app in your browser. Internet access is needed only for a first-time dependency installation and optional movie-poster lookups; recommendations use the bundled data offline.

Keep the launcher window open while using the app. Press **Ctrl+C** in that window to stop CineMatch. Run the same batch file again to restart it; you do not need to activate the virtual environment or remember a port. If startup reports an error, keep the window open so you can read the message.

To start CineMatch manually from PowerShell instead:

```powershell
cd "C:\path\to\movie_recommender_nlp"
.\.venv\Scripts\python.exe launch_app.py
```

The launcher script can also be started from the project folder with `python launch_app.py` after dependencies are installed. It automatically chooses an available port.

```bash
# Open a terminal in the project folder
python -m venv .venv

# Windows PowerShell
.venv\\Scripts\\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 4. Run the offline demo

The repository includes an offline synthetic dataset so the app and evaluation can run without downloading anything.

```bash
python scripts/generate_demo_data.py   # optional: regenerate sample data
python run_demo.py                     # print recommendations and explanations
python run_evaluation.py               # compute ranking metrics and create graphs
streamlit run app.py                   # open the interactive application
```

### Movie poster artwork

The app loads recommendation posters from matching Wikipedia film articles. Poster lookups are cached across runs, and the optional movie-catalog posters load only when requested so initial page loads stay quick. No API key or account is required; built-in fallback artwork is shown when a matching poster is unavailable. Poster copyrights remain with their respective owners.

Generated outputs are written to `results/`:

- `evaluation_metrics.csv` - metrics for the four ranking methods.
- `evaluation_results.json` - dataset and metric summary.
- `per_user_ranks.csv` - held-out test rank for each user and method.
- `ranking_metrics.png`, `map_precision.png`, `latent_intents.png` - report/demo graphs.
- `sample_recommendations.csv` - one sample user's top recommendations.

## 5. Dataset

### Included demo dataset

- 50 movie records with title, year, genres, short project-written overview, and tags.
- 90 synthetic user profiles and 3,060 synthetic rating events.
- Ratings are generated from latent genre-preference vectors with controlled noise.
- **These ratings are synthetic. They are not real MovieLens ratings and do not support real-world effectiveness claims.**

The report's numerical results are from this synthetic offline dataset. This is clearly labelled so the demo remains reproducible without a network connection.

### Optional MovieLens 100K data

An optional converter is supplied for the public GroupLens MovieLens 100K dataset:

```bash
python scripts/download_movielens.py
```

It downloads the GroupLens archive and creates `data/ml-100k/movies.csv` and `data/ml-100k/ratings.csv`. Then set the data directory before running the app or evaluation.

Windows PowerShell:

```powershell
$env:MOVIE_DATA_DIR = "data/ml-100k"
python run_evaluation.py
streamlit run app.py
```

macOS/Linux:

```bash
MOVIE_DATA_DIR=data/ml-100k python run_evaluation.py
MOVIE_DATA_DIR=data/ml-100k streamlit run app.py
```

MovieLens 100K includes user ratings and genre flags but not movie plot descriptions in `u.item`. The downloader creates short genre-derived text so the NLP pipeline can run; richer plot/review text would improve dimension extraction. MovieLens use is optional and needs internet access.

## 6. Methodology

### Pre-processing and text enrichment

1. Lowercase and normalize punctuation/whitespace.
2. Detect aspect dimensions from lexical cues and map movie genres to semantically related dimensions.
3. Build one combined document: `clean overview + genres + dimensions`.
4. Generate TF-IDF unigram/bigram features for content retrieval.
5. Generate a count-vector representation for NMF latent-topic discovery.
6. Convert ratings `>= 4` to positive BPR events and ratings `<= 2` to negative events; neutral ratings are not used in pairwise updates.

### Latent intent and ranking

- NMF is fitted to movie text and produces a topic distribution per movie.
- A user's topic profile is the rating-weighted average topic distribution of positively rated movies.
- A softmax attention layer gives more weight to stronger latent intents.
- BPR learns user/movie factors using pairwise positive-vs-negative comparisons.
- The hybrid score is `0.20 * BPR + 0.65 * content + 0.15 * intent`, after per-user candidate-set min-max normalization.

### Personalized explanation

The explanation generator selects dimensions from the user's positive-rating history and the recommended movie's extracted dimensions. It then produces a user/movie-specific sentence. It does not use an LLM or a trained natural-language generator.

## 7. Evaluation protocol and metrics

- **Split:** leave-one-positive-out. For each eligible user, the latest positive rating (`rating >= 4`) is held out; all other interactions are used for training.
- **Candidates:** movies not present in that user's training history. The held-out positive is one relevant candidate. Unobserved movies are treated as non-relevant for this offline ranking calculation; in real implicit-feedback studies, unobserved does not always mean disliked.
- **Baselines:** global positive-rating popularity, TF-IDF content-only, and BPR-only.
- **Proposed model:** hybrid BPR + content + attentive topic score.
- **Metrics:** HR@K, Recall@K, NDCG@K, MAP@10, and Precision@10. With exactly one held-out relevant item per user, HR@K and Recall@K have the same numerical value.

Run `python run_evaluation.py` to reproduce the outputs. The report includes the current run's actual values. Results may vary slightly with dependency versions or changed random seeds.

## 8. Project files

```text
movie_recommender_nlp/
├── app.py
├── run_demo.py
├── run_evaluation.py
├── requirements.txt
├── README.md
├── AI_AND_EXTERNAL_CODE_DECLARATION.md
├── REFERENCES.md
├── src/
│   ├── data.py
│   ├── evaluation.py
│   ├── model.py
│   ├── nlp_features.py
│   └── recommender.py
├── scripts/
│   ├── generate_demo_data.py
│   └── download_movielens.py
├── data/demo/
├── results/
└── report/
    ├── Mini_Project_Report.docx
    └── Mini_Project_Report.pdf
```

## 9. Limitations and future work

- The included ratings are synthetic and genre-driven; benchmark conclusions require evaluation on real MovieLens or another appropriately licensed dataset.
- The dimension extractor is a manually designed lexicon, not contextual LLM inference. It may miss implicit aspects or map ambiguous words incorrectly.
- NMF is a lightweight topic-model proxy, not a variational neural topic model (VAE/NTM) as in ANTM.
- The explanation component uses grounded templates rather than the personalized Transformer architecture and context-prediction objective in PETER.
- Leave-one-positive-out evaluation with unobserved candidates can create false negatives and does not model temporal drift, exposure bias, or cold-start users.
- Future work: use real movie reviews/plot synopses, human-annotated aspect labels, a neural topic model, a learned personalized explanation generator, temporal splits, and significance tests across multiple random seeds.

## 10. Demonstration guide

See [`DEMO_GUIDE.md`](DEMO_GUIDE.md) for a suggested demonstration sequence, expected metrics, commands, and final submission checks. The formal report is provided as [`Mini_Project_Report.docx`](report/Mini_Project_Report.docx) and [`Mini_Project_Report.pdf`](report/Mini_Project_Report.pdf).

## 11. Academic-use reminder

Edit group/member fields in the report, verify the implementation on your lab machines, and make sure every group member can explain the data split, text dimensions, NMF attention, BPR updates, metric calculations, and limitations. The project includes an AI-tool declaration; update it to accurately reflect your final use.
