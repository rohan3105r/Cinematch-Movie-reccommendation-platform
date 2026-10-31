"""Streamlit UI for the Explainable Personalized Movie Recommender."""
from __future__ import annotations

import html
import os
import re
from concurrent.futures import ThreadPoolExecutor
from html import escape

import requests
import streamlit as st

from src.data import load_data
from src.recommender import MovieRecommender

st.set_page_config(page_title="CineMatch NLP", page_icon="🎬", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.35rem 0 1rem;
        margin-bottom: 0.25rem;
    }
    .top-nav-brand {
        color: #fdba74 !important;
        font-size: 1.05rem;
        font-weight: 850;
        letter-spacing: 0.08em;
        text-decoration: none !important;
        white-space: nowrap;
    }
    .top-nav-links {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        flex-wrap: nowrap;
        gap: 0.1rem;
        white-space: nowrap;
    }
    .top-nav-links a {
        display: inline-block;
        color: #cbd5e1 !important;
        text-decoration: none !important;
        font-size: 0.78rem;
        font-weight: 650;
        padding: 0.42rem 0.48rem;
        border-radius: 0.55rem;
        transition: background 140ms ease, color 140ms ease;
    }
    .top-nav-links a:hover {
        background: rgba(249, 115, 22, 0.15);
        color: #fdba74 !important;
    }
    @media (max-width: 640px) {
        .top-nav {
            gap: 0.5rem;
        }
        .top-nav-links {
            justify-content: flex-start;
            gap: 0;
        }
        .top-nav-links a {
            padding: 0.35rem 0.38rem;
            font-size: 0.72rem;
        }
    }
    .stApp {
        background: #090d16;
        color: #f8fafc;
    }
    .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3,
    .stApp [data-testid="stMarkdownContainer"] {
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #263244;
    }
    [data-testid="stSelectbox"] [role="combobox"],
    [data-testid="stSelectbox"] [data-baseweb="select"],
    [data-testid="stSelectbox"] [role="option"] {
        cursor: pointer !important;
    }
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 35%, #7c2d12 100%);
        border: 1px solid rgba(251, 146, 60, 0.25);
        border-radius: 1.2rem;
        padding: 1.5rem 1.5rem 1.2rem;
        box-shadow: 0 20px 45px rgba(15, 23, 42, 0.18);
        margin-bottom: 1.1rem;
        color: white;
    }
    .section-tag {
        display: inline-block;
        background: rgba(251, 146, 60, 0.16);
        border: 1px solid rgba(253, 186, 116, 0.55);
        border-radius: 999px;
        padding: 0.28rem 0.8rem;
        font-size: 0.75rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
        font-weight: 700;
        color: #fdba74;
    }
    .kpi-card {
        background: linear-gradient(180deg, #151d2b 0%, #101722 100%);
        border: 1px solid #263244;
        border-radius: 1rem;
        padding: 1rem 1.1rem;
        height: 100%;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
    }
    .kpi-label {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 0.3rem;
    }
    .taste-pill {
        display: inline-block;
        background: rgba(249, 115, 22, 0.12);
        border: 1px solid rgba(251, 146, 60, 0.35);
        border-radius: 999px;
        padding: 0.28rem 0.8rem;
        margin: 0.2rem 0.35rem 0.2rem 0;
        font-size: 0.8rem;
        font-weight: 600;
        color: #fdba74;
    }
    .profile-panel {
        background: #151d2b;
        border: 1px solid #263244;
        border-radius: 1rem;
        padding: 1rem;
        backdrop-filter: blur(4px);
    }
    .profile-panel h3 {
        color: white;
        margin-top: 0.1rem;
        margin-bottom: 0.4rem;
    }
    .profile-panel .muted {
        color: rgba(255,255,255,0.8);
    }
    .score-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.32rem 0.65rem;
        border-radius: 0.35rem;
        background: #f5c518;
        color: #111827;
        font-size: 0.92rem;
        font-weight: 800;
        white-space: nowrap;
    }
    .score-caption {
        color: #94a3b8;
        font-size: 0.72rem;
        margin-left: 0.4rem;
    }
    .muted {
        color: #94a3b8;
    }
    .streamlit-expanderHeader {
        background: #151d2b;
        border-radius: 0.75rem;
        color: #f8fafc;
        font-weight: 700;
    }
    .catalog-title {
        color: #f8fafc;
        font-weight: 750;
        font-size: 1.02rem;
        line-height: 1.25;
        min-height: 2.6rem;
        margin-top: 0.55rem;
        overflow-wrap: anywhere;
    }
    .catalog-meta {
        color: #94a3b8;
        font-size: 0.78rem;
        line-height: 1.4;
        min-height: 2.2rem;
    }
    .poster-fallback {
        width: 100%;
        max-width: 160px;
        aspect-ratio: 2 / 3;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        padding: 1rem;
        border-radius: 0.75rem;
        background: linear-gradient(145deg, #7c2d12, #1e293b 55%, #0f172a);
        color: white;
        font-weight: 800;
        box-shadow: inset 0 0 35px rgba(0,0,0,0.35);
        overflow-wrap: anywhere;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Building NLP and recommendation models...")
def get_model(data_dir: str):
    movies, ratings = load_data(data_dir if data_dir else None)
    model = MovieRecommender().fit(movies, ratings)
    return movies, ratings, model


@st.cache_data(show_spinner=False, persist="disk")
def get_poster(title: str, year: int | None) -> tuple[str | None, str | None]:
    headers = {"User-Agent": "CineMatchNLP/1.0 (educational movie discovery app)"}
    page_names = [f"{title.replace(' ', '_')}_({year}_film)"] if year else []
    page_names.append(title.replace(" ", "_"))
    try:
        for page_name in page_names:
            response = requests.get(
                f"https://en.wikipedia.org/wiki/{requests.utils.quote(page_name, safe='()_:-')}",
                headers=headers,
                timeout=(4, 8),
                stream=True,
            )
            with response:
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                page_start = ""
                for chunk in response.iter_content(chunk_size=32768):
                    page_start += chunk.decode(response.encoding or "utf-8", errors="replace")
                    image_row = re.search(
                        r'<td[^>]*class="[^"]*\binfobox-image\b[^"]*"[^>]*>.*?</td>',
                        page_start,
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                    if image_row:
                        image = re.search(r'<img[^>]+src="([^"]+)"', image_row.group(), flags=re.IGNORECASE)
                        if image:
                            image_url = html.unescape(image.group(1))
                            if image_url.startswith("//"):
                                image_url = f"https:{image_url}"
                            return image_url, None
                    if len(page_start) > 512_000:
                        break
    except requests.RequestException as exc:
        status = exc.response.status_code if exc.response is not None else exc.__class__.__name__
        return None, f"Wikipedia movie artwork lookup failed ({status})."

    return None, None


def fetch_posters(movie_rows) -> tuple[dict[int, str], list[str]]:
    records = movie_rows.to_dict("records")

    def lookup(record):
        year_value = record.get("year")
        year = int(year_value) if year_value is not None and str(year_value).strip() not in ("", "nan") else None
        url, error = get_poster(str(record["title"]), year)
        return int(record["movie_id"]), url, error

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(lookup, records))
    posters = {movie_id: url for movie_id, url, _ in results if url}
    errors = list(dict.fromkeys(error for _, _, error in results if error))
    return posters, errors


def show_poster(movie, posters: dict[int, str]):
    movie_id = int(movie["movie_id"])
    poster_url = posters.get(movie_id)
    if poster_url:
        st.image(poster_url, caption=str(movie["title"]), width=160)
    else:
        title = escape(str(movie["title"]))
        st.markdown(
            f"<div class='poster-fallback'><span>🎬</span><strong>{title}</strong></div>",
            unsafe_allow_html=True,
        )


configured_dir = os.environ.get("MOVIE_DATA_DIR", "")
try:
    movies, ratings, recommender = get_model(configured_dir)
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.info("Run `python scripts/generate_demo_data.py` from the project folder and restart the app.")
    st.stop()

users = sorted(ratings.user_id.unique().astype(int).tolist())


def user_profile(user_id: int):
    movie_table = getattr(recommender, "movies", movies)
    history = ratings[ratings.user_id == user_id].merge(
        movie_table[["movie_id", "title", "genres", "dimensions"]],
        on="movie_id",
        how="left",
    )
    positive = history[history.rating >= 4].copy()
    if positive.empty:
        return {
            "positive_count": 0,
            "avg_rating": 0.0,
            "top_genres": [],
            "top_dimensions": [],
        }

    genre_counts = positive["genres"].fillna("").str.split("|").explode().str.strip()
    genre_counts = genre_counts[genre_counts.ne("")].value_counts()
    top_genres = genre_counts.head(3).index.tolist()

    dims = positive["dimensions"].fillna("").str.split("|").explode()
    dims = dims[dims.ne("")].value_counts()
    top_dimensions = dims.head(4).index.tolist()

    return {
        "positive_count": len(positive),
        "avg_rating": round(float(positive.rating.mean()), 2),
        "top_genres": top_genres,
        "top_dimensions": top_dimensions,
    }


with st.sidebar:
    st.header("Recommendation controls")
    selected_user = st.selectbox(
        "Choose a user",
        users,
        index=0,
        help="Click to choose a different user and refresh their personalized recommendations.",
    )
    model_name = st.selectbox(
        "Ranking model",
        ["hybrid", "bpr", "content", "intent", "popularity"],
        index=0,
        help="Hybrid combines BPR, content similarity, and latent intent signals.",
    )
    top_k = st.slider("Movie count", min_value=5, max_value=15, value=10)
    hide_seen = st.checkbox("Hide movies already rated", value=True)
    st.divider()
    st.caption("Demo ratings are synthetic. To use MovieLens 100K instead, set MOVIE_DATA_DIR and run the downloader from the project folder.")

st.markdown(
    """
    <nav class="top-nav" aria-label="Main navigation">
        <a class="top-nav-brand" href="#home">CINEMATCH</a>
        <div class="top-nav-links">
            <a href="#recommendations">For You</a>
            <a href="#explore">Explore</a>
            <a href="#taste">My Taste</a>
            <a href="#about">About</a>
        </div>
    </nav>
    <div id="home"></div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='app-header'>
        <div class='section-tag'>CineMatch</div>
        <h1 style='margin:0; font-size:2.5rem; line-height:1.15;'>Movie recommendations with reason, not guesswork</h1>
        <p style='margin:0.7rem 0 0 0; color: rgba(255,255,255,0.9); max-width: 900px;'>Explore a cinematic recommendation feed shaped by user taste, genre affinity, latent intent, and explainable movie dimensions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

summary = user_profile(selected_user)

c1, c2, c3, c4 = st.columns(4)
c1.markdown(
    "<div class='kpi-card'><div class='kpi-label'>Movies</div><div class='kpi-value'>%s</div></div>" % f"{len(movies):,}",
    unsafe_allow_html=True,
)
c2.markdown(
    "<div class='kpi-card'><div class='kpi-label'>Users</div><div class='kpi-value'>%s</div></div>" % f"{ratings.user_id.nunique():,}",
    unsafe_allow_html=True,
)
c3.markdown(
    "<div class='kpi-card'><div class='kpi-label'>Liked</div><div class='kpi-value'>%s</div></div>" % summary["positive_count"],
    unsafe_allow_html=True,
)
c4.markdown(
    "<div class='kpi-card'><div class='kpi-label'>Avg</div><div class='kpi-value'>%s</div></div>" % summary["avg_rating"],
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='profile-panel'><span class='muted'>Taste spotlight for User %s</span><div style='margin-top:0.5rem;'>%s</div></div>"
    % (
        selected_user,
        "".join(f"<span class='taste-pill'>{escape(g)}</span>" for g in summary["top_genres"][:3])
        if summary["top_genres"]
        else "<span class='muted'>No preference signals yet</span>",
    ),
    unsafe_allow_html=True,
)

st.markdown("<div id='taste'></div>", unsafe_allow_html=True)
st.markdown("<hr style='margin: 1.2rem 0 1rem; border: 0; border-top: 1px solid rgba(148,163,184,0.25);' />", unsafe_allow_html=True)
st.subheader(f"Taste profile for User {selected_user}")
col_left, col_right = st.columns([1.7, 2.3])
with col_left:
    st.markdown("**Favorite genres**")
    if summary["top_genres"]:
        for genre in summary["top_genres"]:
            st.markdown(f"<span class='taste-pill'>{genre}</span>", unsafe_allow_html=True)
    else:
        st.info("This user has no positive ratings yet.")

with col_right:
    st.markdown("**Top dimensions**")
    if summary["top_dimensions"]:
        for dim in summary["top_dimensions"]:
            st.markdown(f"<span class='taste-pill'>{dim}</span>", unsafe_allow_html=True)
    else:
        st.info("No dimensions have been learned yet for this user.")

st.markdown("<div id='recommendations'></div>", unsafe_allow_html=True)
st.subheader(f"Top {top_k} recommendations for User {selected_user}")
recs = recommender.recommend(selected_user, k=top_k, algorithm=model_name, exclude_seen=hide_seen)
if recs.empty:
    st.warning("No unseen movies are available for this user. Try turning off the hide-seen filter or selecting another user.")
else:
    rec_poster_rows = recommender.movies[
        recommender.movies.movie_id.isin(recs.movie_id)
    ][["movie_id", "title", "year"]]
    with st.spinner("Loading posters for your recommendations..."):
        rec_posters, poster_errors = fetch_posters(rec_poster_rows)
    if poster_errors:
        st.warning(poster_errors[0] + " Showing fallback artwork where needed.")
    recommendation_columns = st.columns(2)
    for position, (_, movie) in enumerate(recs.iterrows()):
        with recommendation_columns[position % 2]:
            with st.container(border=True):
                poster_movie = recommender.movie_lookup.loc[int(movie.movie_id)]
                show_poster(poster_movie, rec_posters)
                match_rating = max(0.0, min(10.0, float(movie.score) * 10))
                st.markdown(
                    f"<span class='score-badge'>★ {match_rating:.1f} / 10</span>"
                    "<span class='score-caption'>CineMatch match</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"<div class='catalog-title'>{escape(str(movie.title))}</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='catalog-meta'>{escape(str(movie.genres).replace('|', ' · '))}</div>",
                    unsafe_allow_html=True,
                )
                with st.expander("Why this match?"):
                    st.write(movie.explanation)

st.markdown("<hr style='margin: 1.5rem 0 1rem; border: 0; border-top: 1px solid #263244;' />", unsafe_allow_html=True)
st.markdown("<div id='explore'></div>", unsafe_allow_html=True)
st.subheader("Explore the full movie collection")
filter_col, genre_col = st.columns([2.2, 1])
with filter_col:
    search_query = st.text_input("Search movies", placeholder="Search by title, genre, or story...")
with genre_col:
    genres = sorted(
        {
            genre.strip()
            for value in movies["genres"].fillna("")
            for genre in str(value).split("|")
            if genre.strip()
        }
    )
    selected_genre = st.selectbox("Filter by genre", ["All genres", *genres])

catalog = movies.copy()
if search_query.strip():
    query = search_query.strip()
    matches = (
        catalog["title"].fillna("").str.contains(query, case=False, regex=False)
        | catalog["genres"].fillna("").str.contains(query, case=False, regex=False)
        | catalog["overview"].fillna("").str.contains(query, case=False, regex=False)
    )
    catalog = catalog[matches]
if selected_genre != "All genres":
    catalog = catalog[catalog["genres"].fillna("").str.split("|").apply(lambda values: selected_genre in values)]
catalog = catalog.sort_values("title").reset_index(drop=True)

if catalog.empty:
    st.info("No movies match those filters. Try another title or genre.")
else:
    page_size = 12
    page_count = (len(catalog) + page_size - 1) // page_size
    page_col, count_col = st.columns([1, 4])
    with page_col:
        current_page = st.select_slider("Page", options=list(range(1, page_count + 1)))
    with count_col:
        st.caption(f"Showing {min(page_size, len(catalog) - (current_page - 1) * page_size)} of {len(catalog)} movies")

    start = (current_page - 1) * page_size
    visible_movies = catalog.iloc[start:start + page_size]
    load_catalog_posters = st.checkbox(
        "Load posters for catalog",
        value=False,
        help="Recommendation posters load automatically. Load catalog artwork only when you want to browse posters here.",
    )
    catalog_posters = {}
    if load_catalog_posters:
        with st.spinner("Loading catalog posters..."):
            catalog_posters, catalog_errors = fetch_posters(visible_movies)
        if catalog_errors:
            st.caption("Some poster requests could not be completed; fallback artwork is shown.")

    catalog_columns = st.columns(3)
    for position, (_, movie) in enumerate(visible_movies.iterrows()):
        with catalog_columns[position % 3]:
            with st.container(border=True):
                show_poster(movie, catalog_posters)
                year = int(movie.year) if "year" in movie and str(movie.year) not in ("nan", "") else ""
                st.markdown(f"<div class='catalog-title'>{escape(str(movie.title))}</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='catalog-meta'>{escape(str(year))} · {escape(str(movie.genres).replace('|', ' · '))}</div>",
                    unsafe_allow_html=True,
                )

st.caption("Movie posters are retrieved from Wikipedia article thumbnails via Wikimedia. Copyright remains with the respective poster owners.")

with st.expander("View this user's rating history"):
    history = ratings[ratings.user_id == selected_user].merge(
        movies[["movie_id", "title", "genres"]], on="movie_id", how="left"
    )
    history = history.sort_values("timestamp", ascending=False)
    st.dataframe(history[["title", "genres", "rating", "timestamp"]], use_container_width=True, hide_index=True)

st.markdown("<div id='about'></div>", unsafe_allow_html=True)
with st.expander("How the recommender works"):
    st.markdown(
        """
        1. **Text enrichment:** movie descriptions and genres are turned into transparent dimensions like story, acting, visuals, music, suspense, and emotion.
        2. **Content matching:** TF-IDF compares the user's liked movies with each candidate movie.
        3. **Latent user intent:** NMF topic patterns capture underlying movie themes, and attention weights the topics most relevant to this user.
        4. **Collaborative signal:** BPR matrix factorization learns pairwise user-movie preferences from positive and negative ratings.
        5. **Personalized explanation:** the app combines the user’s top dimensions with the recommended movie’s dimensions to create a clear, grounded reason.

        This is a lightweight, explainable mini-project inspired by the research papers, not a full reproduction of their original architectures.
        """
    )

st.caption("For evaluation plots and metrics, run `python run_evaluation.py` from the project folder.")
