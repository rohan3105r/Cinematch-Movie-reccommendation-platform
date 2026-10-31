"""Build the project report DOCX from the current evaluation outputs."""
from __future__ import annotations
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.section import WD_SECTION

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
RESULTS_DIR = ROOT / "results"
REPORT_DIR.mkdir(exist_ok=True)

NAVY = "17365D"
BLUE = "2F75B5"
LIGHT = "D9EAF7"
PALE = "EEF4F9"
DARK = "243447"


def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tcPr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_text(cell, text, bold=False, color=None, size=8.5):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(str(text))
    r.bold = bold
    r.font.size = Pt(size)
    if color:
        r.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def style_table(table, header=True, font_size=8.2):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for ri, row in enumerate(table.rows):
        tr_pr = row._tr.get_or_add_trPr()
        cant_split = OxmlElement("w:cantSplit")
        tr_pr.append(cant_split)
        if ri == 0 and header:
            repeat_header = OxmlElement("w:tblHeader")
            repeat_header.set(qn("w:val"), "true")
            tr_pr.append(repeat_header)
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.space_before = Pt(2)
                for r in p.runs:
                    r.font.size = Pt(font_size)
            if ri == 0 and header:
                shade(cell, NAVY)
                for p in cell.paragraphs:
                    for r in p.runs:
                        r.bold = True
                        r.font.color.rgb = RGBColor(255, 255, 255)


def add_table(doc, headers, rows, widths=None, font_size=8.2):
    table = doc.add_table(rows=1, cols=len(headers))
    for i, h in enumerate(headers):
        set_cell_text(table.rows[0].cells[i], h, bold=True, color="FFFFFF", size=font_size)
        shade(table.rows[0].cells[i], NAVY)
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            set_cell_text(cells[i], value, size=font_size)
            if len(table.rows) % 2 == 0:
                shade(cells[i], PALE)
    if widths:
        for row in table.rows:
            for cell, width in zip(row.cells, widths):
                cell.width = Inches(width)
    style_table(table, header=True, font_size=font_size)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    p.add_run(text)
    return p


def add_number(doc, text):
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.space_after = Pt(3)
    p.add_run(text)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(8)
    r.font.color.rgb = RGBColor.from_string("666666")


def build_architecture():
    fig, ax = plt.subplots(figsize=(11, 4.3))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.3)
    ax.axis("off")
    boxes = [
        (0.2, 2.35, 2.1, 1.2, "Movie metadata\n+ user ratings"),
        (2.65, 2.35, 2.2, 1.2, "NLP preprocessing\n+ dimension mining"),
        (5.2, 2.35, 2.2, 1.2, "TF-IDF content\n+ NMF latent intents"),
        (7.75, 2.35, 2.8, 1.2, "BPR ranking\n+ intent attention"),
        (3.55, 0.35, 3.2, 1.15, "Hybrid top-K ranking\n+ personalized explanation"),
    ]
    for x, y, w, h, label in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12", facecolor="#EAF2F8", edgecolor="#2F75B5", linewidth=1.5)
        ax.add_patch(patch)
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=10, color="#17365D", weight="bold")
    arrows = [((2.3, 2.95), (2.65, 2.95)), ((4.85, 2.95), (5.2, 2.95)), ((7.4, 2.95), (7.75, 2.95)), ((9.15, 2.35), (6.5, 1.5)), ((5.2, 2.35), (4.8, 1.5))]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.3, color="#5B6573", connectionstyle="arc3,rad=0.0"))
    fig.tight_layout()
    path = RESULTS_DIR / "project_architecture.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def configure_doc(doc):
    sec = doc.sections[0]
    sec.page_width = Inches(8.27)
    sec.page_height = Inches(11.69)
    sec.top_margin = Inches(0.72)
    sec.bottom_margin = Inches(0.7)
    sec.left_margin = Inches(0.78)
    sec.right_margin = Inches(0.78)
    styles = doc.styles
    styles["Normal"].font.name = "Aptos"
    styles["Normal"].font.size = Pt(9.3)
    styles["Normal"].font.color.rgb = RGBColor.from_string(DARK)
    styles["Normal"].paragraph_format.space_after = Pt(5)
    for name, size in [("Heading 1", 16), ("Heading 2", 12), ("Heading 3", 10.5)]:
        st = styles[name]
        st.font.name = "Aptos Display"
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor.from_string(NAVY if name == "Heading 1" else BLUE)
        st.paragraph_format.space_before = Pt(10)
        st.paragraph_format.space_after = Pt(4)
    header = sec.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    rr = header.add_run("CINEMATCH NLP  |  MINI PROJECT")
    rr.font.size = Pt(8)
    rr.font.color.rgb = RGBColor.from_string("718096")
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rr = footer.add_run("K. J. Somaiya School of Engineering  •  Natural Language Processing")
    rr.font.size = Pt(7.5)
    rr.font.color.rgb = RGBColor.from_string("718096")


def main():
    metrics = pd.read_csv(RESULTS_DIR / "evaluation_metrics.csv")
    summary = json.loads((RESULTS_DIR / "evaluation_results.json").read_text(encoding="utf-8"))
    arch_path = build_architecture()
    doc = Document()
    configure_doc(doc)

    # Cover page
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(62)
    r = p.add_run("NATURAL LANGUAGE PROCESSING")
    r.bold = True; r.font.size = Pt(12); r.font.color.rgb = RGBColor.from_string(BLUE)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(14)
    r = p.add_run("CineMatch")
    r.bold = True; r.font.size = Pt(34); r.font.color.rgb = RGBColor.from_string(NAVY)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Explainable Personalized Movie Recommendation using\nText-Dimension Enrichment, Latent Intent Attention and Pairwise Ranking")
    r.font.size = Pt(17); r.font.color.rgb = RGBColor.from_string(DARK)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(16)
    r = p.add_run("MINI-PROJECT REPORT")
    r.bold = True; r.font.size = Pt(11); r.font.color.rgb = RGBColor.from_string(BLUE)

    doc.add_paragraph(" ")
    add_table(doc, ["Academic details", "Information"], [
        ["University", "Somaiya Vidyavihar University"],
        ["Institute", "K. J. Somaiya School of Engineering, Mumbai"],
        ["Programme / Year", "FY M.Tech - Computer Engineering"],
        ["Course", "Natural Language Processing (316P07E123)"],
        ["Academic year / Semester", "2026-2027 / I"],
        ["Faculty in-charge", "Dr. Grishma Sharma"],
        ["Group number", "[Enter group number]"],
    ], widths=[2.1, 4.5], font_size=9)
    add_table(doc, ["Group member", "Name", "Roll number"], [
        ["Member 1", "Adnan Shaikh", "16030726008"],
        ["Member 2", "Aaryan Raut", "16030726018"],
        ["Member 3", "Rohan Rahalkar", "16030726011"],
    ], widths=[2.1, 2.4, 2.1], font_size=9)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(15)
    r = p.add_run("Submitted in partial fulfilment of the Natural Language Processing mini-project")
    r.italic = True; r.font.size = Pt(8.5)
    doc.add_page_break()

    # Contents and abstract
    doc.add_heading("Contents", level=1)
    for item in [
        "Abstract", "1. Introduction and problem definition", "2. Research foundation and comparative analysis",
        "3. Dataset and data pre-processing", "4. Algorithms and techniques", "5. Implementation",
        "6. Evaluation protocol and metrics", "7. Results and discussion", "8. Limitations and future work",
        "9. Conclusion", "References", "AI tools and external code declaration", "Group contribution statement"
    ]:
        add_bullet(doc, item)

    doc.add_heading("Abstract", level=1)
    doc.add_paragraph(
        "This project develops CineMatch, an explainable personalized movie recommendation application using natural-language processing and collaborative ranking. The system enriches movie descriptions with interpretable dimensions, represents movie text using TF-IDF, discovers latent topics with Non-negative Matrix Factorization (NMF), weights user-specific topics with a softmax attention mechanism, and learns pairwise user-item preferences using Bayesian Personalized Ranking (BPR). A hybrid score combines collaborative, content, and latent-intent signals. Each recommendation is accompanied by a short explanation grounded in the user's positively rated movie dimensions and the candidate movie's extracted dimensions. The design is inspired by the supplied PRR-LI, PETER, and ANTM papers, with lighter student-project implementations in place of their full LLM, Transformer, and variational neural-topic architectures. The included offline experiment uses 50 movie records, 90 synthetic users, and 3,060 synthetic rating events. The hybrid model achieves HR@5 of 0.444; content-only retrieval achieves HR@10 of 0.744 and NDCG@10 of 0.391. These are reproducible demonstration results on synthetic data and are not real-world benchmark claims."
    )

    doc.add_heading("1. Introduction and problem definition", level=1)
    doc.add_paragraph(
        "Large movie catalogs make it difficult for viewers to identify films aligned with their interests. A rating-only collaborative model can learn user-item patterns, but it may not explain why a title is recommended. A text-only model can use film descriptions and genres, but it may not capture patterns in user interactions. The project addresses both issues by combining textual movie understanding, user-specific latent intents, pairwise ranking, and feature-grounded natural-language explanations."
    )
    doc.add_heading("1.1 Problem statement", level=2)
    doc.add_paragraph(
        "Given a set of movies M, user rating histories R, and a target user u, rank movies not yet rated by u according to estimated relevance. The system should also generate an explanation E(u,m) for each recommended movie m, using evidence from the user's prior positive ratings and the movie's text-derived dimensions."
    )
    doc.add_heading("1.2 Objectives", level=2)
    for x in [
        "Pre-process movie descriptions and extract interpretable dimensions such as story, acting, visuals, music, suspense, romance, emotion, and social themes.",
        "Build TF-IDF content representations and discover latent movie topics from bag-of-words features.",
        "Learn user-item ranking preferences with BPR and user-specific topic attention.",
        "Produce personalized explanations and evaluate ranking performance against popularity, content-only, and BPR-only baselines.",
    ]:
        add_bullet(doc, x)

    # Research foundation
    doc.add_heading("2. Research foundation and comparative analysis", level=1)
    doc.add_paragraph("The three supplied papers address complementary parts of an explainable personalized recommendation pipeline. Their ideas are adapted as follows.")
    add_table(doc, ["Paper", "Problem / method in the paper", "Strength / limitation relevant to this project", "Project adaptation"], [
        ["Xu & Xu (2024), PRR-LI [1]", "Mines explicit and implicit review dimensions using an LLM, rewrites text, and encodes `text + dimension` together. Evaluates with Recall@N, MAP@N, and NDCG@N.", "Strength: makes implicit attributes available to the matching model. Limitation: depends on LLM dimension mining and domain-specific text processing.", "Rule-based aspect lexicon and genre mapping; combines overview, genres, and dimensions before TF-IDF. This is transparent and offline but less context-aware than an LLM."],
        ["Li et al. (2021), PETER [2]", "Uses user/item identifiers and context prediction to personalize Transformer-generated explanations while also predicting ratings/recommendations. Reports text quality and explainability metrics, including BLEU/ROUGE and feature-oriented measures.", "Strength: user and item both influence explanations. Limitation: Transformer training and reference explanations require additional data and resources.", "Generates a user-item-specific template explanation from observed preference dimensions and candidate movie dimensions. It is grounded and lightweight, but not a trained Transformer."],
        ["Zhang et al. (2025), ANTM [3]", "Uses a neural topic model to infer latent user intents, attention to distinguish their importance, and BPR for personalized item ranking. Reports HR and NDCG.", "Strength: combines latent intent discovery, attention, and pairwise preference ranking. Limitation: neural topic inference and model training are more complex than a compact classroom implementation.", "Uses NMF topic distributions as a topic-model proxy, softmax attention over user topic preference, and BPR matrix factorization. NMF is not the paper's VAE-based neural topic model."],
    ], widths=[1.1, 2.0, 1.9, 1.9], font_size=7.3)
    doc.add_heading("2.1 Integrated research gap", level=2)
    doc.add_paragraph(
        "The combined project treats text dimensions as additional semantic evidence, uses the user's interaction history to weight latent interests, and makes the recommendation explanation dependent on both the user and the movie. The implementation intentionally prioritizes transparency and reproducibility over architectural fidelity. It does not claim to reproduce PRR-LI, PETER, or ANTM."
    )

    # Data
    doc.add_heading("3. Dataset and data pre-processing", level=1)
    add_table(doc, ["Dataset property", "Value in the included offline run"], [
        ["Movie catalog", f"{summary['movie_count']} movies"],
        ["Users", f"{summary['user_count']} synthetic users"],
        ["Rating events", f"{summary['rating_count']:,} synthetic ratings"],
        ["Training interactions", f"{summary['train_interactions']:,}"],
        ["Held-out positives", f"{summary['held_out_positive_interactions']:,} (one per eligible user)"],
        ["Positive / negative threshold", "Positive: rating >= 4; negative: rating <= 2; rating 3 ignored in BPR updates"],
    ], widths=[2.25, 4.3], font_size=8.5)
    doc.add_paragraph(
        "The packaged demo catalog contains 50 movie records with title, year, genres, and short project-written descriptions. The 3,060 rating events were generated synthetically from user genre-preference vectors with noise. They are not ratings collected from real users and are not MovieLens records. This provides a self-contained demonstration, not a substitute for evaluation on a public benchmark. An optional script downloads and converts MovieLens 100K for a real-data experiment."
    )
    doc.add_heading("3.1 Pre-processing steps", level=2)
    for x in [
        "Text normalization: lowercase, remove punctuation except useful apostrophes/hyphens, and normalize whitespace.",
        "Dimension extraction: scan movie overview/tags for aspect cues and map available genres to semantic dimensions.",
        "Combined encoding: concatenate normalized overview text, genre names, and dimension labels as `text + dimension` input.",
        "TF-IDF: generate unigram and bigram vectors for content similarity.",
        "Bag of words: generate count vectors for NMF topic discovery.",
        "Interaction labels: treat ratings >= 4 as positive and ratings <= 2 as negative for pairwise BPR; neutral ratings are not used in pairwise updates.",
    ]:
        add_number(doc, x)
    doc.add_paragraph(
        "Evaluation uses leave-one-positive-out: the latest positive interaction for each eligible user is held out, and the remaining interactions train the models. Candidate items are those absent from that user's training history. In the offline ranking calculation, unobserved candidates are treated as non-relevant; this assumption can introduce false negatives because an unobserved movie is not necessarily disliked."
    )

    # Methods
    doc.add_heading("4. Algorithms and techniques", level=1)
    doc.add_heading("4.1 Dimension-enriched NLP representation", level=2)
    doc.add_paragraph(
        "A manually curated cue lexicon identifies dimensions including story and plot, acting and characters, visuals and cinematography, music and soundtrack, action and stunts, humour and comedy, emotion and relationships, world and imagination, mystery and suspense, romance, and social themes. Genre mapping supplies additional semantic cues. The resulting dimension labels are appended to the normalized movie text and jointly represented by TF-IDF."
    )
    doc.add_heading("4.2 TF-IDF content profile", level=2)
    doc.add_paragraph(
        "The movie documents are vectorized with TF-IDF unigram and bigram features. A user profile is computed as a rating-weighted average of vectors for movies rated 4 or 5. The cosine similarity between this profile and a candidate movie vector gives a content relevance score."
    )
    doc.add_heading("4.3 NMF latent intents and attention", level=2)
    doc.add_paragraph(
        "NMF factorizes the movie-document count matrix into non-negative movie-topic weights and topic-word weights. The user's topic profile is a rating-weighted average of topic distributions for positively rated movies. A softmax over the topic profile produces attention weights; each candidate's intent score is the dot product between its topic distribution and the user's attention weights. This is an interpretable topic-model proxy inspired by ANTM, not a variational autoencoder or neural topic model."
    )
    doc.add_heading("4.4 Bayesian Personalized Ranking", level=2)
    doc.add_paragraph(
        "BPR learns user and item latent vectors from pairwise comparisons. For a user u, positive item i, and negative item j, it seeks a larger score for i than j by optimizing the logistic pairwise objective -log(sigmoid(s(u,i)-s(u,j))) with L2 regularization. The implementation uses 32 latent factors, 18 epochs, learning rate 0.025, regularization 0.01, and seed 42 by default."
    )
    doc.add_heading("4.5 Hybrid score and explanation", level=2)
    doc.add_paragraph(
        "Each component is min-max normalized over the candidate set. The project hybrid score is 0.20 x BPR + 0.65 x content + 0.15 x attentive intent. The weights were selected as transparent demonstration hyperparameters and are not taken from the papers. The explanation generator uses the user's most frequent positive-history dimensions and the candidate's extracted dimensions to produce a sentence tied to that user-item pair. It uses templates rather than an LLM or a trained Transformer."
    )
    doc.add_picture(str(arch_path), width=Inches(6.6))
    add_caption(doc, "Figure 1. CineMatch processing and recommendation pipeline.")

    # Implementation
    doc.add_heading("5. Implementation", level=1)
    add_table(doc, ["File / module", "Responsibility"], [
        ["app.py", "Streamlit interface: select a demo user, ranking model, top-K, and seen-item filter; display ranked movies and explanations."],
        ["src/nlp_features.py", "Text cleaning, lexicon-based dimension extraction, and `text + dimension` enrichment."],
        ["src/model.py", "BPR matrix factorization, TF-IDF profiles, NMF topics, and softmax intent attention."],
        ["src/recommender.py", "Fit the pipeline, compute hybrid/baseline scores, and generate personalized explanations."],
        ["src/evaluation.py", "Leave-one-positive-out evaluation and top-K ranking metrics."],
        ["run_evaluation.py", "Run experiments and create CSV, JSON, and PNG result artifacts."],
        ["scripts/generate_demo_data.py", "Generate the offline synthetic movie catalog and rating events."],
        ["scripts/download_movielens.py", "Optional MovieLens 100K downloader and converter."],
    ], widths=[2.0, 4.6], font_size=8)
    doc.add_heading("5.1 How to execute", level=2)
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.18)
    r = p.add_run("pip install -r requirements.txt\npython scripts/generate_demo_data.py\npython run_demo.py\npython run_evaluation.py\nstreamlit run app.py")
    r.font.name = "Consolas"; r.font.size = Pt(8.5)
    doc.add_paragraph("The first four commands build data, print a terminal recommendation example, and regenerate the evaluation outputs. The final command launches the interactive application in a browser.")

    # Keep the metric definition table together instead of splitting a row over pages.
    doc.add_page_break()
    # Evaluation
    doc.add_heading("6. Evaluation protocol and metrics", level=1)
    doc.add_paragraph(
        f"A leave-one-positive-out split produced {summary['held_out_positive_interactions']} held-out positive interactions for {summary['user_count']} evaluated users. Each user has one relevant test movie. The candidate set consists of movies not rated in the training history. Four methods are compared: global popularity, TF-IDF content-only, BPR-only, and the proposed hybrid."
    )
    add_table(doc, ["Metric", "Meaning in this project"], [
        ["HR@K", "Fraction of users whose held-out positive movie appears within the top K."],
        ["Recall@K", "Fraction of the user's held-out relevant set retrieved in top K. With one relevant item per user, it equals HR@K."],
        ["NDCG@K", "Discounted gain that rewards placing the relevant movie near the top of the ranking."],
        ["MAP@10", "Mean reciprocal rank contribution for the single relevant test item, truncated at 10."],
        ["Precision@10", "Number of relevant results in top 10 divided by 10; at most one relevant item is held out per user."],
        ["Explanation USR", "Unique Sentence Ratio across one top-1 explanation per evaluated user. It measures textual variation, not perceived explanation quality."],
    ], widths=[1.35, 5.2], font_size=8.1)

    # Start result tables and figures as a new section/page for readability.
    doc.add_page_break()
    # Results
    doc.add_heading("7. Results and discussion", level=1)
    table_rows = []
    for _, row in metrics.iterrows():
        table_rows.append([
            str(row["Method"]).upper(), f"{row['HR@5']:.3f}", f"{row['HR@10']:.3f}",
            f"{row['NDCG@5']:.3f}", f"{row['NDCG@10']:.3f}",
            f"{row['MAP@10']:.3f}", f"{row['Precision@10']:.3f}"
        ])
    add_table(doc, ["Method", "HR@5", "HR@10", "NDCG@5", "NDCG@10", "MAP@10", "P@10"], table_rows,
              widths=[1.25, .78, .78, .86, .9, .82, .75], font_size=8)
    doc.add_paragraph(
        f"The hybrid model reaches HR@5 = {metrics.loc[metrics.Method == 'hybrid', 'HR@5'].iloc[0]:.3f} and NDCG@5 = {metrics.loc[metrics.Method == 'hybrid', 'NDCG@5'].iloc[0]:.3f}, the highest top-5 values among the compared methods in this run. At K=10, content-only is strongest (HR@10 = {metrics.loc[metrics.Method == 'content', 'HR@10'].iloc[0]:.3f}; NDCG@10 = {metrics.loc[metrics.Method == 'content', 'NDCG@10'].iloc[0]:.3f}; MAP@10 = {metrics.loc[metrics.Method == 'content', 'MAP@10'].iloc[0]:.3f}). The hybrid remains above BPR-only and popularity on HR@10 and NDCG@10, but does not beat content-only at those cutoffs. This indicates that the textual movie features are highly predictive in this genre-driven synthetic dataset, while collaborative and latent-intent signals improve the early top-five ranking for the current weights."
    )
    doc.add_picture(str(RESULTS_DIR / "ranking_metrics.png"), width=Inches(6.3))
    add_caption(doc, "Figure 2. HR and NDCG comparison at K=5 and K=10.")
    doc.add_picture(str(RESULTS_DIR / "map_precision.png"), width=Inches(6.0))
    add_caption(doc, "Figure 3. MAP@10 and Precision@10 comparison.")
    doc.add_picture(str(RESULTS_DIR / "latent_intents.png"), width=Inches(6.1))
    add_caption(doc, "Figure 4. Top terms describing NMF latent topics.")
    doc.add_paragraph(
        f"The explanation Unique Sentence Ratio is {summary['explanation_unique_sentence_ratio']:.3f} across one top-1 explanation per evaluated user. This indicates that the user-item template generator does not collapse all users to one identical sentence. It does not establish naturalness or usefulness; a human evaluation would be required for those claims. BLEU/ROUGE are not reported because this project has no human-authored reference explanation for each user-movie pair."
    )
    doc.add_heading("7.1 Key observations", level=2)
    for x in [
        "Content-only retrieval provides a strong baseline because the synthetic rating generator is explicitly based on latent genre preferences, which are also present in the movie text.",
        "The hybrid model has the strongest HR@5 and NDCG@5 in this run, suggesting a benefit from combining content with learned collaborative and latent-intent signals near the top of the list.",
        "BPR-only is better than popularity on the reported ranking metrics, showing that user-specific pairwise learning provides value beyond global item frequency in this sample.",
        "The results are descriptive only. There is one split and synthetic data; no statistical significance test or real-world generalization claim is made.",
    ]:
        add_bullet(doc, x)

    # Limits and conclusion
    doc.add_heading("8. Limitations and future work", level=1)
    for x in [
        "Synthetic ratings are convenient for a reproducible offline demo but cannot validate effectiveness on real users. Run the optional MovieLens experiment and report it separately before making benchmark claims.",
        "The lexicon-based aspect extractor is less context-sensitive than LLM-based implicit dimension mining and may miss indirect movie attributes.",
        "NMF is not the variational neural topic model used by ANTM, and the attention calculation is intentionally lightweight.",
        "Template explanations are feature-grounded but less expressive than PETER's personalized Transformer generation; USR alone does not measure explanation quality.",
        "Unobserved candidate items are treated as non-relevant, and a single held-out interaction per user may be sensitive to the split. Future evaluation should include temporal splits, multiple seeds, confidence intervals, and human explanation ratings.",
    ]:
        add_bullet(doc, x)
    doc.add_heading("9. Conclusion", level=1)
    doc.add_paragraph(
        "CineMatch demonstrates an end-to-end NLP recommendation application that links movie text, interpretable dimensions, latent intents, pairwise user-item learning, and personalized explanations. The project turns complementary ideas from PRR-LI, PETER, and ANTM into a compact, runnable system with a Streamlit interface and reproducible ranking evaluation. In the current synthetic experiment, the hybrid approach improves top-five retrieval over the individual baselines, while content-only remains stronger at top ten. The next essential step is validation on a real movie-interaction dataset and qualitative evaluation of explanation usefulness."
    )

    # References
    doc.add_heading("References", level=1)
    refs = [
        "[1] Bei Xu and Yifan Xu. 2024. Personalized Review Recommendation based on Implicit Dimension Mining. Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pp. 86-91. Association for Computational Linguistics. https://aclanthology.org/2024.naacl-short.8/",
        "[2] Lei Li, Yongfeng Zhang, and Li Chen. 2021. Personalized Transformer for Explainable Recommendation. Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Long Papers), ACL-IJCNLP 2021. https://aclanthology.org/2021.acl-long.383/ (preprint: https://arxiv.org/abs/2105.11601)",
        "[3] Yixin Zhang, Sichen Lin, Zhili Zhao, Xuran Zhu, and Chenbo He. 2025. Personalized recommendation by integrating a neural topic model and Bayesian personalized ranking. Knowledge-Based Systems, 311, 113110. https://doi.org/10.1016/j.knosys.2025.113110",
        "[4] F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems, 5(4), Article 19. https://doi.org/10.1145/2827872",
        "[5] scikit-learn developers. scikit-learn: Machine Learning in Python. https://scikit-learn.org/",
        "[6] Streamlit developers. Streamlit documentation. https://docs.streamlit.io/",
    ]
    for ref in refs:
        p = doc.add_paragraph(ref)
        p.paragraph_format.left_indent = Inches(0.18)
        p.paragraph_format.first_line_indent = Inches(-0.18)
        p.paragraph_format.space_after = Pt(5)

    doc.add_heading("AI tools and external code declaration", level=1)
    doc.add_paragraph(
        "AI tool declaration: OpenAI ChatGPT (GPT-6) was used as an AI assistant to interpret the uploaded assignment and research papers, help structure the project, draft/organize the source code, and prepare report text. The core recommendation and evaluation code was executed and unit-tested in the provided environment. The Streamlit interface is included but must be launched after installing the packages in requirements.txt. All submitting students must review the implementation, independently understand and test it, and ensure this declaration matches the final submitted work and institutional policy."
    )
    doc.add_paragraph(
        "External code declaration: No third-party project repository or codebase was copied. The source files were written for this mini-project. The software uses open-source Python packages listed in requirements.txt (NumPy, pandas, scikit-learn, Matplotlib, Streamlit, and python-docx). The optional MovieLens utility uses Python standard-library networking and ZIP modules."
    )
    doc.add_paragraph(
        "Data declaration: Included ratings are synthetic and generated by scripts/generate_demo_data.py. They are not real user ratings or MovieLens observations. If the optional MovieLens downloader is used, GroupLens MovieLens 100K must be acknowledged as the dataset source."
    )
    doc.add_heading("Group contribution statement", level=1)
    add_table(doc, ["Member", "Name / roll number", "Contribution (complete accurately)"], [
        ["Member 1", "Adnan Shaikh / 16030726008", "[To be completed by group]"],
        ["Member 2", "Aaryan Raut / 16030726018", "[To be completed by group]"],
        ["Member 3", "Rohan Rahalkar / 16030726011", "[To be completed by group]"],
    ], widths=[1.0, 2.2, 3.3], font_size=7.8)
    doc.add_paragraph("The member names and roll numbers are carried over from the uploaded Activity 1 report. The actual division of work and signatures must be completed and verified by all group members before submission.")

    out = REPORT_DIR / "Mini_Project_Report.docx"
    doc.save(out)
    print(out)


if __name__ == "__main__":
    main()
