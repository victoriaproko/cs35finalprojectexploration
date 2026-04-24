from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


CSV_PATH = Path("high_popularity_spotify_data.csv")
LEARNING_PATH = Path("artist_learning.json")
COLS_OF_INTEREST = [
    "energy",
    "tempo",
    "danceability",
    "playlist_genre",
    "loudness",
    "liveness",
    "valence",
    "track_artist",
    "time_signature",
    "speechiness",
    "track_popularity",
    "track_album_name",
    "playlist_name",
    "track_name",
    "track_album_release_date",
    "instrumentalness",
    "mode",
    "key",
    "duration_ms",
    "acousticness",
    "playlist_subgenre",
]
FINAL_COLS = [
    "energy",
    "tempo",
    "danceability",
    "playlist_genre",
    "loudness",
    "liveness",
    "valence",
    "speechiness",
    "instrumentalness",
    "acousticness",
    "playlist_subgenre",
    "artist_list",
]
ANSWER_MAP = {
    "y": 1.0,
    "yes": 1.0,
    "p": 0.8,
    "probably": 0.8,
    "u": 0.5,
    "unknown": 0.5,
    "idk": 0.5,
    "?": 0.5,
    "pn": 0.2,
    "probably not": 0.2,
    "n": 0.0,
    "no": 0.0,
}
NUMERIC_QUESTION_TEMPLATES = {
    "energy": ("high energy", "low energy"),
    "danceability": ("very danceable", "not very danceable"),
    "tempo": ("fast tempo", "slow tempo"),
    "valence": ("upbeat", "moody"),
    "speechiness": ("speech-heavy", "not speech-heavy"),
    "acousticness": ("acoustic", "not acoustic"),
    "instrumentalness": ("often instrumental", "rarely instrumental"),
    "liveness": ("live-sounding", "not live-sounding"),
    "loudness": ("loud", "soft"),
}
MIDDLE_VALUE_QUESTION_TEMPLATES = {
    "tempo": "mid-tempo",
}
GENRE_LABELS = {
    "hip-hop": "hip-hop/rap",
    "latin": "Latin",
    "pop": "pop",
    "rock": "rock",
    "r&b": "R&B",
    "electronic": "electronic",
    "ambient": "ambient",
    "afrobeats": "Afrobeats",
    "gaming": "gaming",
    "arabic": "Arabic",
}
SUBGENRE_LABELS = {
    "gangster": "gangster rap",
    "modern": "modern pop",
    "global": "global crossover",
    "mainstream": "mainstream pop",
    "throwback": "throwback hits",
    "reggaeton": "reggaeton",
    "trap": "trap",
    "melodic": "melodic rap",
    "alternative": "alternative",
    "drill": "drill",
    "nigerian": "Nigerian Afrobeats",
    "techno": "techno",
    "afro-latin": "Afro-Latin",
    "african": "African",
    "academic": "academic",
    "meditative": "meditative",
    "soft": "soft pop",
    "classic": "classic hits",
}
EXCLUDED_SUBGENRE_QUESTIONS = {"chill"}


@dataclass(frozen=True)
class QuestionSpec:
    key: str
    text: str


def load_dataset_final(csv_path: Path | str = CSV_PATH, min_songs: int = 10) -> pd.DataFrame:
    """Recreate the notebook's dataset_final from the raw Spotify CSV."""
    dataset = pd.read_csv(csv_path)
    dataset_cleaned = dataset[COLS_OF_INTEREST].copy()

    dataset_cleaned["artist_list"] = dataset_cleaned["track_artist"].str.split(",")
    dataset_exploded = dataset_cleaned.explode("artist_list")
    dataset_exploded["artist_list"] = dataset_exploded["artist_list"].str.strip()
    dataset_exploded["song_count"] = dataset_exploded.groupby("artist_list")["artist_list"].transform("size")

    name_fixes = {
        "Tyler": "Tyler, the Creator",
        "The Creator": "Tyler, the Creator",
    }
    dataset_exploded["artist_list"] = dataset_exploded["artist_list"].replace(name_fixes)
    dataset_exploded["artist_list"] = dataset_exploded["artist_list"].str.strip()

    bad_names = ["Tyler", "the Creator"]
    dataset_exploded = dataset_exploded[~dataset_exploded["artist_list"].isin(bad_names)]

    dataset_final = dataset_exploded[dataset_exploded["song_count"] >= min_songs].copy()
    return dataset_final[FINAL_COLS].reset_index(drop=True)


def entropy(beliefs: Dict[str, float]) -> float:
    return -sum(prob * math.log2(prob) for prob in beliefs.values() if prob > 0)


class SpotifyArtistGuesser:
    """Bayesian artist guesser built directly from the project's dataset_final."""

    def __init__(
        self,
        kb: Dict[str, Dict[str, float]],
        questions: List[QuestionSpec],
        artist_song_counts: Dict[str, int],
        learning_path: Path | str = LEARNING_PATH,
    ) -> None:
        self.base_kb = {artist: values.copy() for artist, values in kb.items()}
        self.kb = {artist: values.copy() for artist, values in kb.items()}
        self.questions = questions
        self.question_text = {question.key: question.text for question in questions}
        self.artists = sorted(kb)
        self.artist_song_counts = artist_song_counts
        self.learning_path = Path(learning_path)
        self.learning_stats = self._load_learning_stats()
        self._apply_learning_stats()

    @classmethod
    def from_dataset(
        cls,
        dataset_final: pd.DataFrame,
        alpha: float = 1.0,
        min_genre_songs: int = 12,
        min_subgenre_songs: int = 3,
        learning_path: Path | str = LEARNING_PATH,
    ) -> "SpotifyArtistGuesser":
        if "artist_list" not in dataset_final.columns:
            raise ValueError("dataset_final must contain an 'artist_list' column.")

        artist_counts = dataset_final["artist_list"].value_counts().sort_index()
        artists = artist_counts.index.tolist()
        questions: List[QuestionSpec] = []
        kb: Dict[str, Dict[str, float]] = {artist: {} for artist in artists}

        def add_question(question: QuestionSpec, yes_counts: pd.Series) -> None:
            aligned_counts = yes_counts.reindex(artists, fill_value=0.0).astype(float)
            smoothed = (aligned_counts + alpha) / (artist_counts.astype(float) + (2.0 * alpha))
            questions.append(question)
            for artist, probability in smoothed.items():
                kb[artist][question.key] = float(probability)

        def add_probability_question(question: QuestionSpec, probabilities: pd.Series) -> None:
            aligned = probabilities.reindex(artists, fill_value=0.0).astype(float).clip(0.0, 1.0)
            questions.append(question)
            for artist, probability in aligned.items():
                kb[artist][question.key] = float(probability)

        def add_scaled_mean_question(
            question: QuestionSpec,
            values: pd.Series,
            center: float,
            reverse: bool = False,
        ) -> None:
            aligned = values.reindex(artists).astype(float)
            iqr = float(aligned.quantile(0.75) - aligned.quantile(0.25))
            std = float(aligned.std())
            scale = max(iqr / 2.0, std / 2.0, 1e-6)
            if reverse:
                logits = (center - aligned) / scale
            else:
                logits = (aligned - center) / scale
            probabilities = 1.0 / (1.0 + (-logits).apply(math.exp))
            add_probability_question(question, probabilities)

        def add_middle_mean_question(question: QuestionSpec, values: pd.Series, center: float) -> None:
            aligned = values.reindex(artists).astype(float)
            iqr = float(aligned.quantile(0.75) - aligned.quantile(0.25))
            std = float(aligned.std())
            scale = max(iqr / 3.0, std / 3.0, 1e-6)
            z_scores = (aligned - center) / scale
            probabilities = (-0.5 * (z_scores ** 2)).apply(math.exp).clip(0.02, 0.98)
            add_probability_question(question, probabilities)

        numeric_columns = [
            "energy",
            "danceability",
            "tempo",
            "valence",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "loudness",
        ]
        artist_means = dataset_final.groupby("artist_list")[numeric_columns].mean().reindex(artists)

        for column in numeric_columns:
            mean_values = artist_means[column]
            low_threshold = float(mean_values.quantile(0.33))
            middle_threshold = float(mean_values.quantile(0.50))
            high_threshold = float(mean_values.quantile(0.67))
            high_label, low_label = NUMERIC_QUESTION_TEMPLATES[column]

            add_scaled_mean_question(
                QuestionSpec(
                    key=f"{column}_high",
                    text=f"On average, are this artist's songs {high_label}?",
                ),
                mean_values,
                high_threshold,
            )

            add_scaled_mean_question(
                QuestionSpec(
                    key=f"{column}_low",
                    text=f"On average, are this artist's songs {low_label}?",
                ),
                mean_values,
                low_threshold,
                reverse=True,
            )

            if column in MIDDLE_VALUE_QUESTION_TEMPLATES:
                middle_label = MIDDLE_VALUE_QUESTION_TEMPLATES[column]
                add_middle_mean_question(
                    QuestionSpec(
                        key=f"{column}_mid",
                        text=f"On average, are this artist's songs {middle_label}?",
                    ),
                    mean_values,
                    middle_threshold,
                )

        genre_counts = dataset_final["playlist_genre"].value_counts()
        active_genres = genre_counts[genre_counts >= min_genre_songs].index.tolist()
        genre_share = pd.crosstab(
            dataset_final["artist_list"],
            dataset_final["playlist_genre"],
            normalize="index",
        ).reindex(index=artists, fill_value=0.0)
        dominant_genre = genre_share.idxmax(axis=1)

        for genre in active_genres:
            genre_label = GENRE_LABELS.get(genre, genre.replace("-", " "))
            genre_mask = dataset_final["playlist_genre"] == genre
            genre_yes_counts = genre_mask.groupby(dataset_final["artist_list"]).sum()
            add_probability_question(
                QuestionSpec(
                    key=f"genre_{genre}_share",
                    text=f"Is this artist mostly {genre_label}?",
                ),
                genre_share[genre] if genre in genre_share.columns else pd.Series(0.0, index=artists),
            )
            add_probability_question(
                QuestionSpec(
                    key=f"genre_{genre}_dominant",
                    text=f"Is {genre_label} their main genre?",
                ),
                dominant_genre.eq(genre).astype(float).replace({1.0: 0.98, 0.0: 0.02}),
            )

        subgenre_counts = dataset_final["playlist_subgenre"].value_counts()
        active_subgenres = [
            subgenre
            for subgenre in subgenre_counts[subgenre_counts >= min_subgenre_songs].index.tolist()
            if subgenre not in EXCLUDED_SUBGENRE_QUESTIONS
        ]
        subgenre_share = pd.crosstab(
            dataset_final["artist_list"],
            dataset_final["playlist_subgenre"],
            normalize="index",
        ).reindex(index=artists, fill_value=0.0)
        dominant_subgenre = subgenre_share.idxmax(axis=1)
        for subgenre in active_subgenres:
            subgenre_label = SUBGENRE_LABELS.get(subgenre, subgenre.replace("-", " "))
            add_probability_question(
                QuestionSpec(
                    key=f"subgenre_{subgenre}",
                    text=f"Is this artist associated with {subgenre_label}?",
                ),
                subgenre_share[subgenre] if subgenre in subgenre_share.columns else pd.Series(0.0, index=artists),
            )
            add_probability_question(
                QuestionSpec(
                    key=f"subgenre_{subgenre}_dominant",
                    text=f"Is {subgenre_label} their main subgenre?",
                ),
                dominant_subgenre.eq(subgenre).astype(float).replace({1.0: 0.97, 0.0: 0.03}),
            )

        return cls(
            kb=kb,
            questions=questions,
            artist_song_counts=artist_counts.to_dict(),
            learning_path=learning_path,
        )

    def initial_beliefs(self) -> Dict[str, float]:
        prior = 1.0 / len(self.artists)
        return {artist: prior for artist in self.artists}

    def _load_learning_stats(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        if not self.learning_path.exists():
            return {}

        try:
            raw = json.loads(self.learning_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

        cleaned: Dict[str, Dict[str, Dict[str, float]]] = {}
        for artist, question_map in raw.items():
            if artist not in self.base_kb or not isinstance(question_map, dict):
                continue
            cleaned[artist] = {}
            for question_key, stat in question_map.items():
                if question_key not in self.base_kb[artist] or not isinstance(stat, dict):
                    continue
                count = int(stat.get("count", 0))
                total = float(stat.get("sum", 0.0))
                if count > 0:
                    cleaned[artist][question_key] = {"count": count, "sum": total}
        return cleaned

    def _save_learning_stats(self) -> None:
        if not self.learning_stats:
            return
        self.learning_path.write_text(
            json.dumps(self.learning_stats, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _apply_learning_stats(self) -> None:
        self.kb = {artist: values.copy() for artist, values in self.base_kb.items()}

        for artist, question_map in self.learning_stats.items():
            if artist not in self.kb:
                continue
            for question_key, stat in question_map.items():
                if question_key not in self.kb[artist]:
                    continue
                count = int(stat.get("count", 0))
                if count <= 0:
                    continue
                learned_mean = float(stat["sum"]) / count
                base_value = self.base_kb[artist][question_key]
                learned_weight = min(0.75, count / (count + 3.0))
                self.kb[artist][question_key] = ((1.0 - learned_weight) * base_value) + (
                    learned_weight * learned_mean
                )

    def learn_from_session(self, actual_artist: str, answers: Dict[str, float]) -> bool:
        if actual_artist not in self.base_kb:
            return False

        artist_stats = self.learning_stats.setdefault(actual_artist, {})
        for question_key, answer_value in answers.items():
            if question_key not in self.base_kb[actual_artist]:
                continue
            stat = artist_stats.setdefault(question_key, {"count": 0, "sum": 0.0})
            stat["count"] = int(stat["count"]) + 1
            stat["sum"] = float(stat["sum"]) + float(answer_value)

        self._apply_learning_stats()
        self._save_learning_stats()
        return True

    @staticmethod
    def _normalize_beliefs(beliefs: Dict[str, float]) -> Dict[str, float]:
        total = sum(beliefs.values())
        if total <= 0:
            if not beliefs:
                return {}
            uniform = 1.0 / len(beliefs)
            return {artist: uniform for artist in beliefs}
        return {artist: probability / total for artist, probability in beliefs.items()}

    def eliminate_artist(self, beliefs: Dict[str, float], artist: str) -> Dict[str, float]:
        remaining = {name: probability for name, probability in beliefs.items() if name != artist}
        return self._normalize_beliefs(remaining)

    @staticmethod
    def _prompt_yes_no(prompt: str) -> bool:
        while True:
            raw = input(prompt).strip().lower()
            if raw in {"y", "yes"}:
                return True
            if raw in {"n", "no"}:
                return False
            print("  Please answer with y / n")

    def update_beliefs(
        self,
        beliefs: Dict[str, float],
        question_key: str,
        answer_value: float,
        noise: float = 0.1,
    ) -> Dict[str, float]:
        posteriors: Dict[str, float] = {}
        for artist, prior in beliefs.items():
            p_yes = self.kb[artist][question_key]
            likelihood = (p_yes * answer_value) + ((1.0 - p_yes) * (1.0 - answer_value))
            likelihood = ((1.0 - noise) * likelihood) + (noise * 0.5)
            posteriors[artist] = prior * likelihood

        return self._normalize_beliefs(posteriors)

    def expected_info_gain(self, question_key: str, beliefs: Dict[str, float]) -> float:
        p_yes = sum(beliefs[artist] * self.kb[artist][question_key] for artist in beliefs)
        p_no = 1.0 - p_yes
        if p_yes <= 0 or p_no <= 0:
            return 0.0

        h_before = entropy(beliefs)
        post_yes = self.update_beliefs(beliefs, question_key, 1.0, noise=0.05)
        post_no = self.update_beliefs(beliefs, question_key, 0.0, noise=0.05)
        h_after = (p_yes * entropy(post_yes)) + (p_no * entropy(post_no))
        return h_before - h_after

    def pick_question(self, beliefs: Dict[str, float], asked: set[str]) -> Optional[QuestionSpec]:
        remaining = [question for question in self.questions if question.key not in asked]
        if not remaining:
            return None
        return max(remaining, key=lambda question: self.expected_info_gain(question.key, beliefs))

    def top_guesses(self, beliefs: Dict[str, float], limit: int = 3) -> List[Tuple[str, float]]:
        return sorted(beliefs.items(), key=lambda item: item[1], reverse=True)[:limit]

    def explain_artist_profile(self, artist: str, limit: int = 15) -> pd.DataFrame:
        if artist not in self.kb:
            raise ValueError(f"Unknown artist: {artist}")
        rows = [
            {
                "question_key": key,
                "question": self.question_text[key],
                "probability_yes": round(value, 4),
                "distance_from_50_50": round(abs(value - 0.5), 4),
            }
            for key, value in self.kb[artist].items()
        ]
        profile = pd.DataFrame(rows)
        profile = profile.sort_values(
            ["probability_yes", "distance_from_50_50"],
            ascending=[False, False],
        )
        return profile.head(limit)

    def trace_artist_simulation(self, artist: str, steps: int = 10) -> pd.DataFrame:
        if artist not in self.kb:
            raise ValueError(f"Unknown artist: {artist}")

        beliefs = self.initial_beliefs()
        asked: set[str] = set()
        rows = []

        for step in range(1, steps + 1):
            question = self.pick_question(beliefs, asked)
            if question is None:
                break
            asked.add(question.key)
            answer_value = self.kb[artist][question.key]
            beliefs = self.update_beliefs(beliefs, question.key, answer_value)
            leader, leader_prob = self.top_guesses(beliefs, limit=1)[0]
            rows.append(
                {
                    "step": step,
                    "question_key": question.key,
                    "question": question.text,
                    "artist_answer_value": round(answer_value, 4),
                    "leader_after_step": leader,
                    "leader_probability": round(leader_prob, 4),
                }
            )

        return pd.DataFrame(rows)

    def run_session(
        self,
        threshold: float = 0.80,
        min_questions: int = 5,
        max_questions: int = 20,
        keep_learning: bool = True,
    ) -> Tuple[str, Dict[str, float]]:
        beliefs = self.initial_beliefs()
        asked: set[str] = set()
        answers: Dict[str, float] = {}
        eliminated: List[str] = []
        question_number = 1

        print("Think of one of the artists in dataset_final and I'll try to guess them.")
        print("Answers: y / n / p (probably) / pn (probably not) / u (unknown)\n")

        while beliefs:
            asked_question = False
            if question_number <= max_questions:
                question = self.pick_question(beliefs, asked)
                if question is not None:
                    asked.add(question.key)
                    asked_question = True

                    while True:
                        raw = input(f"Q{question_number}: {question.text}  ").strip().lower()
                        if raw in ANSWER_MAP:
                            break
                        print("  Please answer with y / n / p / pn / u")

                    answer_value = ANSWER_MAP[raw]
                    answers[question.key] = answer_value
                    beliefs = self.update_beliefs(beliefs, question.key, answer_value)
                    leader, leader_prob = self.top_guesses(beliefs, limit=1)[0]
                    print(f"   -> Leading guess: {leader} ({leader_prob:.1%})\n")
                    question_number += 1

                    if leader_prob < threshold or len(asked) < min_questions:
                        continue

            final_guess, confidence = self.top_guesses(beliefs, limit=1)[0]
            print(f"My guess: {final_guess} (confidence: {confidence:.1%})")
            print("\nTop 3:")
            for artist, probability in self.top_guesses(beliefs, limit=3):
                print(f"  {artist}: {probability:.1%}")

            if self._prompt_yes_no("Was I right? (y/n)  "):
                if keep_learning:
                    self.learn_from_session(final_guess, answers)
                print("Nice. I'll remember this session for future runs.")
                return final_guess, beliefs

            eliminated.append(final_guess)
            beliefs = self.eliminate_artist(beliefs, final_guess)
            if not beliefs:
                break

            if question_number <= max_questions and len(asked) < len(self.questions):
                if asked_question:
                    print(f"   -> {final_guess} is eliminated. I'll ask another question.\n")
                else:
                    print(f"   -> {final_guess} is eliminated. I'll go back to questions.\n")
                continue

            print(f"   -> {final_guess} is eliminated. Moving to the next best artist.\n")

        actual_artist = input("\nWho was the artist?  ").strip()
        if keep_learning and self.learn_from_session(actual_artist, answers):
            print(f"Thanks. I updated the model for {actual_artist}.")
        elif actual_artist:
            print("I couldn't learn from that answer because the artist is not in this dataset.")

        if not beliefs and eliminated:
            return eliminated[-1], {}
        fallback_guess = eliminated[-1] if eliminated else ""
        return fallback_guess, beliefs

    def simulate_artist(
        self,
        artist: str,
        threshold: float = 0.80,
        min_questions: int = 5,
        max_questions: int = 20,
    ) -> Tuple[str, float, int]:
        beliefs = self.initial_beliefs()
        asked: set[str] = set()

        for question_count in range(1, max_questions + 1):
            question = self.pick_question(beliefs, asked)
            if question is None:
                break
            asked.add(question.key)
            beliefs = self.update_beliefs(beliefs, question.key, self.kb[artist][question.key])

            leader, leader_prob = self.top_guesses(beliefs, limit=1)[0]
            if leader_prob >= threshold and len(asked) >= min_questions:
                return leader, leader_prob, question_count

        leader, leader_prob = self.top_guesses(beliefs, limit=1)[0]
        return leader, leader_prob, len(asked)

    def evaluate_self_consistency(
        self,
        threshold: float = 0.80,
        min_questions: int = 5,
        max_questions: int = 20,
    ) -> pd.DataFrame:
        rows = []
        for artist in self.artists:
            guess, confidence, questions_used = self.simulate_artist(
                artist,
                threshold=threshold,
                min_questions=min_questions,
                max_questions=max_questions,
            )
            rows.append(
                {
                    "target_artist": artist,
                    "model_guess": guess,
                    "correct": guess == artist,
                    "confidence": round(confidence, 4),
                    "questions_used": questions_used,
                }
            )
        return pd.DataFrame(rows).sort_values(["correct", "confidence"], ascending=[True, False])


def build_artist_guesser(
    dataset_final: Optional[pd.DataFrame] = None,
    learning_path: Path | str = LEARNING_PATH,
) -> SpotifyArtistGuesser:
    if dataset_final is None:
        dataset_final = load_dataset_final()
    return SpotifyArtistGuesser.from_dataset(dataset_final, learning_path=learning_path)


def play(
    dataset_final: Optional[pd.DataFrame] = None,
    threshold: float = 0.80,
    min_questions: int = 5,
    max_questions: int = 20,
    keep_learning: bool = True,
) -> Tuple[str, Dict[str, float]]:
    guesser = build_artist_guesser(dataset_final)
    return guesser.run_session(
        threshold=threshold,
        min_questions=min_questions,
        max_questions=max_questions,
        keep_learning=keep_learning,
    )


if __name__ == "__main__":
    guesser = build_artist_guesser()
    report = guesser.evaluate_self_consistency()
    accuracy = report["correct"].mean()
    print(f"Artists in model: {len(guesser.artists)}")
    print(f"Questions in knowledge base: {len(guesser.questions)}")
    print(f"Self-consistency accuracy: {accuracy:.1%}\n")
    print(report.head(10).to_string(index=False))
    print()
    guesser.run_session()
