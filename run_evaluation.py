"""Run the evaluation and regenerate tables/graphs for the mini-project."""
from __future__ import annotations
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from src.data import load_data
from src.evaluation import evaluate

ROOT = Path(__file__).resolve().parent


def main():
    data_dir = os.environ.get("MOVIE_DATA_DIR")
    movies, ratings = load_data(data_dir)
    metrics, users, model, train, test = evaluate(movies, ratings)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    metrics.to_csv(out / "evaluation_metrics.csv", index=False)
    users.to_csv(out / "per_user_ranks.csv", index=False)
    # PETER-inspired qualitative diversity check: explanations should depend on
    # the user-item pair rather than collapsing to one repeated sentence.
    explanation_samples = []
    for user_id in sorted(test.user_id.unique().astype(int).tolist()):
        top_one = model.recommend(user_id, k=1, algorithm="hybrid", exclude_seen=True)
        if not top_one.empty:
            explanation_samples.append(str(top_one.iloc[0]["explanation"]))
    explanation_usr = (len(set(explanation_samples)) / len(explanation_samples)) if explanation_samples else 0.0
    (out / "sample_explanations.txt").write_text("\n\n".join(explanation_samples[:20]), encoding="utf-8")

    summary = {
        "dataset": str(data_dir or "data/demo"),
        "movie_count": int(len(movies)),
        "rating_count": int(len(ratings)),
        "user_count": int(ratings.user_id.nunique()),
        "train_interactions": int(len(train)),
        "held_out_positive_interactions": int(len(test)),
        "positive_threshold": 4.0,
        "explanation_unique_sentence_ratio": float(explanation_usr),
        "explanation_note": "Unique Sentence Ratio (USR) is computed over one top-1 explanation per evaluated user; it measures variation, not human-perceived quality.",
        "metrics": metrics.to_dict(orient="records"),
        "note": "Demo data are synthetic. Unobserved candidate items are treated as non-relevant for ranking evaluation."
    }
    (out / "evaluation_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Figure 1: ranking quality at K=5 and K=10.
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    x = range(len(metrics))
    ax.plot(list(x), metrics["HR@5"], marker="o", label="HR@5")
    ax.plot(list(x), metrics["HR@10"], marker="o", label="HR@10")
    ax.plot(list(x), metrics["NDCG@5"], marker="s", label="NDCG@5")
    ax.plot(list(x), metrics["NDCG@10"], marker="s", label="NDCG@10")
    ax.set_xticks(list(x), metrics["Method"].str.upper())
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric value")
    ax.set_title("Top-K recommendation quality by method")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out / "ranking_metrics.png", dpi=180)
    plt.close(fig)

    # Figure 2: MAP and precision comparison.
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    metrics.set_index("Method")[["MAP@10", "Precision@10"]].plot(kind="bar", ax=ax)
    ax.set_title("MAP@10 and Precision@10 comparison")
    ax.set_ylabel("Metric value")
    ax.set_xlabel("Recommendation method")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "map_precision.png", dpi=180)
    plt.close(fig)

    # Figure 3: topic word labels from the NMF component weights.
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    topics = model.text_model.topic_terms
    labels = [f"Intent {i+1}: " + ", ".join(words[:4]) for i, words in enumerate(topics)]
    y = list(range(len(labels)))
    ax.barh(y, [len(words) for words in topics])
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Top terms displayed per latent topic")
    ax.set_title("Interpretable latent-intent topic labels (NMF)")
    fig.tight_layout()
    fig.savefig(out / "latent_intents.png", dpi=180)
    plt.close(fig)

    print(f"Explanation Unique Sentence Ratio (USR): {explanation_usr:.3f}")
    print("\nEVALUATION RESULTS")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved graphs and metrics to: {out}")


if __name__ == "__main__":
    main()
