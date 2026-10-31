"""Text preprocessing and interpretable movie-dimension enrichment.

This module implements a transparent lexicon-based approximation of the
implicit-dimension enrichment idea in Xu & Xu (2024). It does not call an LLM.
"""
from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

# Each dimension has lexical cues that may occur explicitly or implicitly in
# a movie overview, tag field, or genre description.
DIMENSION_TERMS: dict[str, tuple[str, ...]] = {
    "story and plot": ("plot", "story", "storyline", "narrative", "screenplay", "twist", "script"),
    "acting and characters": ("acting", "actor", "cast", "performance", "character", "portrayal"),
    "visuals and cinematography": ("visual", "cinematography", "animation", "visual effects", "special effects", "colourful", "colorful", "design"),
    "music and soundtrack": ("music", "soundtrack", "score", "song", "musical", "dance"),
    "action and stunts": ("action", "fight", "chase", "battle", "stunt", "martial arts"),
    "humour and comedy": ("comedy", "comic", "funny", "humor", "humour", "witty", "satire"),
    "emotion and relationships": ("emotional", "heartfelt", "family", "friendship", "relationship", "love", "coming of age"),
    "world and imagination": ("fantasy", "science fiction", "sci-fi", "space", "futuristic", "magical", "universe", "dystopian"),
    "mystery and suspense": ("mystery", "suspense", "thriller", "tension", "detective", "investigation", "crime", "horror"),
    "romance": ("romance", "romantic", "love story", "chemistry"),
    "social themes": ("social", "political", "justice", "inequality", "identity", "historical", "war"),
}

GENRE_DIMENSIONS = {
    "Action": "action and stunts",
    "Adventure": "adventure",
    "Animation": "visuals and cinematography",
    "Children": "family entertainment",
    "Comedy": "humour and comedy",
    "Crime": "mystery and suspense",
    "Documentary": "real-world subjects",
    "Drama": "emotion and relationships",
    "Fantasy": "world and imagination",
    "Film-Noir": "mystery and suspense",
    "Horror": "mystery and suspense",
    "Musical": "music and soundtrack",
    "Mystery": "mystery and suspense",
    "Romance": "romance",
    "Sci-Fi": "world and imagination",
    "Thriller": "mystery and suspense",
    "War": "social themes",
    "Western": "adventure",
}


def clean_text(value: object) -> str:
    """Normalize whitespace and punctuation while retaining useful words."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9\s'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_dimensions(text: str, genres: Iterable[str] | str = ()) -> list[str]:
    """Return explicit/implicit aspect labels using lexical and genre cues."""
    normalized = clean_text(text)
    dims: list[str] = []
    for dimension, terms in DIMENSION_TERMS.items():
        if any(re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", normalized) for term in terms):
            dims.append(dimension)
    if isinstance(genres, str):
        genre_values = [g.strip() for g in genres.replace(",", "|").split("|") if g.strip()]
    else:
        genre_values = [str(g).strip() for g in genres if str(g).strip()]
    for genre in genre_values:
        mapped = GENRE_DIMENSIONS.get(genre)
        if mapped and mapped not in dims:
            dims.append(mapped)
    if not dims:
        dims.append("general storytelling")
    return dims


def enrich_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """Add normalized text, dimension labels, and combined text+dimension input."""
    result = movies.copy()
    if "overview" not in result.columns:
        result["overview"] = ""
    if "genres" not in result.columns:
        result["genres"] = "Drama"
    result["clean_overview"] = result["overview"].map(clean_text)
    result["dimensions_list"] = [
        extract_dimensions(overview, genres)
        for overview, genres in zip(result["clean_overview"], result["genres"])
    ]
    result["dimensions"] = result["dimensions_list"].map(lambda x: "|".join(x))
    # PRR-LI-inspired combined encoding: original text + dimension labels.
    result["combined_text"] = (
        result["clean_overview"].fillna("")
        + " genres " + result["genres"].fillna("").str.lower().str.replace("|", " ", regex=False)
        + " dimensions " + result["dimensions"].str.replace("|", " ", regex=False)
    )
    return result
