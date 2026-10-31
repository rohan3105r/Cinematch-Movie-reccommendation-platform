"""Create a fully offline synthetic movie-interaction dataset for demonstration.

Movie metadata contains public film titles and short original summaries written
for this teaching project. Ratings and user profiles are synthetic, not real
user data and not MovieLens observations.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "demo"

# title, year, genres, concise project-written overview, tags
MOVIES = [
("The Matrix", 1999, "Action|Sci-Fi", "A computer hacker discovers a simulated reality and joins a rebellion that challenges control, identity, and technology.", "cyberpunk technology virtual reality martial arts"),
("Inception", 2010, "Action|Sci-Fi|Thriller", "A skilled team enters layered dreams to perform a complex mission where memory, suspense, and visual spectacle shape the story.", "dreams mind bending heist suspense"),
("Interstellar", 2014, "Adventure|Drama|Sci-Fi", "Explorers travel beyond Earth to search for a future for humanity, combining emotional family relationships with space science and discovery.", "space exploration family emotion science"),
("The Dark Knight", 2008, "Action|Crime|Drama", "A masked hero confronts a criminal mastermind in a tense city conflict about justice, sacrifice, and moral choices.", "superhero crime psychological tension"),
("Toy Story", 1995, "Animation|Adventure|Comedy|Children", "A group of toys experiences friendship, jealousy, and adventure when a new toy changes their owner's world.", "friendship family humor animation"),
("Finding Nemo", 2003, "Animation|Adventure|Children|Comedy", "A cautious father crosses a vast ocean to find his missing child and meets unusual companions along the way.", "family friendship ocean colorful animation"),
("Up", 2009, "Animation|Adventure|Comedy|Drama", "An elderly widower and a young explorer embark on an imaginative journey filled with humor, grief, friendship, and discovery.", "heartfelt adventure colorful family"),
("Coco", 2017, "Animation|Adventure|Children|Drama|Musical", "A young musician enters a vibrant world of ancestors and learns about memory, family bonds, music, and belonging.", "music songs family colorful heartfelt"),
("Inside Out", 2015, "Animation|Children|Comedy|Drama", "Personified emotions guide a child through a major life change in a creative story about identity and emotional growth.", "psychology emotions humor family animation"),
("Spirited Away", 2001, "Animation|Adventure|Fantasy", "A young girl navigates a mysterious spirit world and finds courage through imaginative characters and magical settings.", "fantasy magic world building animation"),
("Parasite", 2019, "Comedy|Drama|Thriller", "Two families become connected through an unexpected arrangement that exposes class divisions, tension, and sharp social commentary.", "social inequality suspense dark humor"),
("The Godfather", 1972, "Crime|Drama", "A powerful crime family navigates loyalty, leadership, and betrayal in a character-driven story about power and morality.", "mafia family loyalty acting"),
("Pulp Fiction", 1994, "Crime|Drama", "Interwoven crime stories use distinctive dialogue, nonlinear storytelling, dark humor, and memorable characters.", "nonlinear narrative dialogue dark comedy"),
("Forrest Gump", 1994, "Comedy|Drama|Romance", "A kind-hearted man experiences major moments in modern history through a life shaped by friendship, love, and perseverance.", "heartfelt romance history humor"),
("The Shawshank Redemption", 1994, "Drama", "A prisoner builds an enduring friendship and holds on to hope while facing injustice inside a harsh institution.", "friendship hope character drama"),
("Good Will Hunting", 1997, "Drama|Romance", "A gifted young man confronts difficult memories and relationships while discovering purpose with the help of a mentor.", "emotional relationships dialogue character"),
("La La Land", 2016, "Comedy|Drama|Musical|Romance", "An aspiring actor and a jazz musician pursue creative ambitions while navigating love, music, and difficult choices.", "music dance romance dreams"),
("Titanic", 1997, "Drama|Romance", "A romance develops aboard a famous ocean liner as class, personal freedom, and disaster shape the characters' lives.", "romantic historical emotional"),
("The Notebook", 2004, "Drama|Romance", "A long relationship is remembered through devotion, separation, and the emotional force of shared memories.", "love story heartfelt relationships"),
("Pride and Prejudice", 2005, "Drama|Romance", "A spirited young woman and a reserved gentleman reconsider first impressions amid family expectations and social conventions.", "period romance social class character"),
("Mad Max: Fury Road", 2015, "Action|Adventure|Sci-Fi|Thriller", "Survivors cross a harsh wasteland in a relentless chase driven by practical stunts, striking visuals, and resistance.", "post apocalyptic chase action cinematography"),
("John Wick", 2014, "Action|Crime|Thriller", "A retired assassin returns to a dangerous underworld, leading to tightly choreographed fights and a revenge-driven story.", "martial arts choreography crime action"),
("Gladiator", 2000, "Action|Adventure|Drama", "A betrayed Roman general seeks justice in an arena while questions of honor and power shape his journey.", "historical battle epic acting"),
("Jurassic Park", 1993, "Action|Adventure|Sci-Fi|Thriller", "A theme park built around cloned dinosaurs becomes a survival adventure when scientific ambition escapes control.", "science adventure suspense visual effects"),
("The Lion King", 1994, "Animation|Adventure|Children|Drama|Musical", "A young lion faces loss and responsibility in a musical coming-of-age story about family and courage.", "songs family identity heartfelt"),
("Avengers: Endgame", 2019, "Action|Adventure|Sci-Fi", "A team of heroes attempts a final mission after a universe-changing conflict, bringing together action and emotional closure.", "superhero teamwork spectacle sacrifice"),
("Black Panther", 2018, "Action|Adventure|Sci-Fi", "A new ruler protects a technologically advanced nation while balancing tradition, identity, and responsibility.", "superhero technology culture action"),
("Spider-Man: Into the Spider-Verse", 2018, "Animation|Action|Adventure|Sci-Fi", "A teenager discovers a new sense of responsibility in a visually inventive multiverse adventure.", "comic animation visuals identity coming of age"),
("Get Out", 2017, "Horror|Mystery|Thriller", "A weekend visit gradually reveals unsettling secrets in a suspenseful story with social themes and psychological horror.", "psychological suspense social commentary"),
("Knives Out", 2019, "Comedy|Crime|Drama|Mystery", "A detective investigates a wealthy family's secrets through clever clues, ensemble performances, and witty dialogue.", "mystery investigation humor ensemble cast"),
("The Grand Budapest Hotel", 2014, "Adventure|Comedy|Drama", "A hotel concierge and his protégé become involved in a caper told through stylized visuals and eccentric characters.", "visual design witty humor adventure"),
("Whiplash", 2014, "Drama|Music", "A young drummer faces an intense instructor in a psychological conflict about ambition, discipline, and musical performance.", "music performance tension character"),
("A Beautiful Mind", 2001, "Drama|Romance", "A mathematical thinker navigates achievement and personal challenges in a relationship-centered biographical drama.", "science mental challenge relationships"),
("The Social Network", 2010, "Drama", "A university project grows into a major technology company amid ambition, conflict, friendship, and legal disputes.", "technology entrepreneurship dialogue"),
("Hidden Figures", 2016, "Drama|History", "Three mathematicians make essential contributions to a space program while challenging discrimination and institutional barriers.", "history science equality inspiration"),
("The Imitation Game", 2014, "Drama|Thriller|War", "A cryptography team works under pressure during wartime, combining historical events, mystery, and human relationships.", "history codebreaking suspense"),
("The Silence of the Lambs", 1991, "Crime|Drama|Thriller", "An investigator seeks insight from an imprisoned killer while pursuing a dangerous criminal in a tense psychological mystery.", "crime investigation suspense acting"),
("Se7en", 1995, "Crime|Mystery|Thriller", "Two detectives follow a disturbing series of crimes through a bleak city and a tightly constructed mystery.", "detective crime dark suspense"),
("The Conjuring", 2013, "Horror|Mystery|Thriller", "Investigators encounter unexplained events in a family home, creating an atmospheric supernatural horror story.", "supernatural mystery suspense"),
("Shutter Island", 2010, "Mystery|Thriller|Drama", "A marshal investigates a disappearance at a remote institution where memory and reality become uncertain.", "psychological mystery twist"),
("The Martian", 2015, "Adventure|Drama|Sci-Fi", "An astronaut uses science, humor, and persistence to survive alone on Mars while a rescue team works from Earth.", "space science survival humor"),
("Dune", 2021, "Adventure|Drama|Sci-Fi", "A young heir enters a desert world shaped by politics, ecology, prophecy, and large-scale science-fiction spectacle.", "world building desert politics science fiction"),
("Arrival", 2016, "Drama|Mystery|Sci-Fi", "A linguist attempts to communicate with visitors from another world in a thoughtful story about language and time.", "language communication science emotional"),
("Blade Runner 2049", 2017, "Drama|Mystery|Sci-Fi|Thriller", "A future investigator uncovers a secret that challenges identity and memory in a visually atmospheric world.", "dystopian identity cinematography mystery"),
("Harry Potter and the Prisoner of Azkaban", 2004, "Adventure|Children|Fantasy|Mystery", "A young wizard returns to school as a mysterious escaped prisoner and hidden family history reshape his world.", "magic friendship mystery fantasy"),
("Pirates of the Caribbean: The Curse of the Black Pearl", 2003, "Action|Adventure|Comedy|Fantasy", "A blacksmith joins an eccentric pirate to rescue a captive and confront a supernatural curse.", "pirates humor adventure fantasy"),
("The Incredibles", 2004, "Action|Adventure|Animation|Children|Comedy", "A family of superheroes balances everyday life and extraordinary action in an animated adventure.", "family teamwork humor animation"),
("Ratatouille", 2007, "Animation|Children|Comedy", "An ambitious rat pursues a passion for cooking in a Paris kitchen through creativity, friendship, and humor.", "food creativity friendship animation"),
("Moana", 2016, "Adventure|Animation|Children|Comedy|Musical", "A determined navigator crosses the ocean to restore balance, discovering courage, culture, and music.", "ocean music identity adventure"),
("Zootopia", 2016, "Animation|Children|Comedy|Crime", "An ambitious rabbit police officer and a fox investigate a city mystery while confronting stereotypes and teamwork.", "mystery social themes humor animation"),
]

GENRES = sorted({g for movie in MOVIES for g in movie[2].split("|")})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    movies = pd.DataFrame([
        {"movie_id": i + 1, "title": title, "year": year, "genres": genres,
         "overview": overview, "tags": tags}
        for i, (title, year, genres, overview, tags) in enumerate(MOVIES)
    ])
    movies.to_csv(OUT / "movies.csv", index=False)

    n_users = 90
    rows = []
    movie_genre_indices = []
    genre_to_idx = {g: i for i, g in enumerate(GENRES)}
    for genres in movies.genres:
        movie_genre_indices.append([genre_to_idx[g] for g in genres.split("|")])

    timestamp_base = 1_600_000_000
    for user_id in range(1, n_users + 1):
        # A Dirichlet profile gives each synthetic user a different mix of tastes.
        taste = rng.dirichlet(np.full(len(GENRES), 0.38))
        sample_size = min(34, len(movies) - 1)
        chosen = rng.choice(len(movies), size=sample_size, replace=False)
        ratings_for_user = []
        for pos, movie_idx in enumerate(chosen):
            gidx = movie_genre_indices[movie_idx]
            # Relative genre preference: 1.0 is roughly average affinity.
            affinity = float(np.mean(taste[gidx]) / (1.0 / len(GENRES)))
            latent_utility = np.clip((affinity - 0.35) / 2.25, 0.0, 1.0)
            noisy = latent_utility + rng.normal(0.0, 0.17)
            rating = int(np.clip(np.rint(1 + 4 * noisy), 1, 5))
            ratings_for_user.append((movie_idx, rating, pos))
        # Ensure each account has enough positive history for a held-out test.
        positive_count = sum(r >= 4 for _, r, _ in ratings_for_user)
        if positive_count < 5:
            ratings_for_user = [(m, max(r, 4) if k < 6 else r, k)
                                for k, (m, r, _) in enumerate(ratings_for_user)]
        rng.shuffle(ratings_for_user)
        for sequence, (movie_idx, rating, _original_pos) in enumerate(ratings_for_user):
            rows.append({
                "user_id": user_id,
                "movie_id": int(movies.iloc[movie_idx].movie_id),
                "rating": int(rating),
                "timestamp": int(timestamp_base + user_id * 100_000 + sequence * 3600),
            })
    ratings = pd.DataFrame(rows)
    ratings.to_csv(OUT / "ratings.csv", index=False)
    print(f"Created {len(movies)} movies and {len(ratings)} synthetic ratings for {n_users} users at {OUT}")
    print("Synthetic interactions are for offline demonstration only; they are not MovieLens or real user data.")


if __name__ == "__main__":
    main()
