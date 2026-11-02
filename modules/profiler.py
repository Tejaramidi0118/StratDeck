"""
Phase 4: Opponent-Style Profiling Module.
Extracts 14 observable behavioral features from game histories, generates
SentenceTransformer text embeddings, performs KMeans clustering with Silhouette score
validation, and produces factual behavioral summaries.
"""

from collections import Counter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

from env.cards import Card, NUM_CARD_TYPES

# 14 Observable Behavioral Feature Names
FEATURE_NAMES = [
    "freq_draw",
    "freq_attack",
    "freq_shield",
    "freq_skip",
    "freq_reverse",
    "freq_spy",
    "freq_swap",
    "freq_trap",
    "freq_steal",
    "offensive_ratio",
    "defensive_ratio",
    "utility_ratio",
    "action_diversity",
    "total_turns",
]


def extract_features(move_history: list) -> np.ndarray:
    """
    Extracts a 14-dimensional observable numerical feature vector from a player's move history.
    move_history: list of (turn_idx, action) tuples.
    """
    if not move_history:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    counts = Counter()
    unique_actions = set()

    for turn_idx, action in move_history:
        act_val = -1 if action == -1 else int(action)
        unique_actions.add(act_val)
        if act_val == -1:
            counts["draw"] += 1
        else:
            counts[act_val] += 1

    total_actions = len(move_history)
    total_turns = max(1, total_actions)

    freq_draw = counts["draw"] / total_actions
    freq_attack = counts[int(Card.ATTACK)] / total_actions
    freq_shield = counts[int(Card.SHIELD)] / total_actions
    freq_skip = counts[int(Card.SKIP)] / total_actions
    freq_reverse = counts[int(Card.REVERSE)] / total_actions
    freq_spy = counts[int(Card.SPY)] / total_actions
    freq_swap = counts[int(Card.SWAP)] / total_actions
    freq_trap = counts[int(Card.TRAP)] / total_actions
    freq_steal = counts[int(Card.STEAL)] / total_actions

    offensive_ratio = freq_attack + freq_trap + freq_steal
    defensive_ratio = freq_shield + freq_skip + freq_reverse
    utility_ratio = freq_spy + freq_swap
    action_diversity = len(unique_actions) / (NUM_CARD_TYPES + 1)  # 10 possible action types

    return np.array([
        freq_draw,
        freq_attack,
        freq_shield,
        freq_skip,
        freq_reverse,
        freq_spy,
        freq_swap,
        freq_trap,
        freq_steal,
        offensive_ratio,
        defensive_ratio,
        utility_ratio,
        action_diversity,
        float(total_turns),
    ], dtype=np.float32)


def feature_to_text(features: np.ndarray) -> str:
    """Converts 14-dim feature vector into a structured textual representation for sentence embedding."""
    f = dict(zip(FEATURE_NAMES, features))
    summary = (
        f"Player Draw Rate: {f['freq_draw']:.1%}, "
        f"Attack Rate: {f['freq_attack']:.1%}, "
        f"Shield Rate: {f['freq_shield']:.1%}, "
        f"Skip Rate: {f['freq_skip']:.1%}, "
        f"Reverse Rate: {f['freq_reverse']:.1%}, "
        f"Spy Rate: {f['freq_spy']:.1%}, "
        f"Swap Rate: {f['freq_swap']:.1%}, "
        f"Trap Rate: {f['freq_trap']:.1%}, "
        f"Steal Rate: {f['freq_steal']:.1%}. "
        f"Overall Ratios -> Offensive: {f['offensive_ratio']:.1%}, "
        f"Defensive: {f['defensive_ratio']:.1%}, "
        f"Utility: {f['utility_ratio']:.1%}. "
        f"Action Diversity: {f['action_diversity']:.1%}, Total Turns: {f['total_turns']:.0f}."
    )
    return summary


class OpponentProfiler:
    def __init__(self, n_clusters: int = None, seed: int = 42):
        self.n_clusters = n_clusters
        self.seed = seed
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2") if _HAS_ST else None
        self.scaler = StandardScaler()
        self.kmeans = None
        self.silhouette_scores = {}
        self._fitted_embeddings = None
        self._fitted_features = None

    def embed_text(self, text: str) -> np.ndarray:
        if self.embedder is not None:
            return self.embedder.encode(text)
        # Fallback bag-of-words hash vector if sentence-transformers unavailable
        return self._hash_embed(text)

    @staticmethod
    def _hash_embed(text: str, dim: int = 384) -> np.ndarray:
        vec = np.zeros(dim, dtype=np.float32)
        for word in text.lower().split():
            vec[hash(word) % dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def fit(self, move_histories: list, min_k: int = 2, max_k: int = 6):
        """
        Fits StandardScaler and KMeans clustering on a list of player move_histories.
        Evaluates silhouette scores for k in [min_k, max_k] if n_clusters is not fixed.
        """
        if not move_histories:
            raise ValueError("Cannot fit OpponentProfiler on empty move_histories list.")

        feature_matrix = np.array([extract_features(h) for h in move_histories], dtype=np.float32)
        self.scaler.fit(feature_matrix)
        norm_features = self.scaler.transform(feature_matrix)

        # Generate MiniLM text embeddings
        text_summaries = [feature_to_text(f) for f in feature_matrix]
        embeddings = np.array([self.embed_text(t) for t in text_summaries], dtype=np.float64)

        self._fitted_features = feature_matrix
        self._fitted_embeddings = embeddings

        # Determine optimal k via Silhouette score if n_clusters not set
        self.silhouette_scores = {}
        best_k = min_k
        best_score = -1.0

        n_samples = len(embeddings)
        valid_max_k = min(max_k, n_samples - 1)

        if valid_max_k < min_k:
            best_k = max(2, min(min_k, n_samples))
        else:
            for k in range(min_k, valid_max_k + 1):
                km = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
                labels = km.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)
                self.silhouette_scores[k] = float(score)
                if score > best_score:
                    best_score = score
                    best_k = k

        chosen_k = self.n_clusters if self.n_clusters is not None else best_k
        self.n_clusters = chosen_k

        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10)
        self.kmeans.fit(embeddings)

        return self

    def predict_cluster(self, move_history: list) -> int:
        """Predicts cluster ID for a given player's move history."""
        if self.kmeans is None:
            return 0
        feat = extract_features(move_history)
        text = feature_to_text(feat)
        emb = np.array(self.embed_text(text), dtype=np.float64)
        cluster_id = int(self.kmeans.predict([emb])[0])
        return cluster_id

    def get_cluster_statistics(self, move_histories: list, labels: list) -> dict:
        """Computes mean and std feature statistics for each cluster."""
        feature_matrix = np.array([extract_features(h) for h in move_histories], dtype=np.float32)
        labels = np.array(labels)
        stats = {}

        for k in range(self.n_clusters):
            idx = np.where(labels == k)[0]
            if len(idx) == 0:
                continue
            c_features = feature_matrix[idx]
            stats[k] = {
                "count": len(idx),
                "percentage": len(idx) / len(labels),
                "means": dict(zip(FEATURE_NAMES, c_features.mean(axis=0).tolist())),
                "stds": dict(zip(FEATURE_NAMES, c_features.std(axis=0).tolist())),
            }
        return stats
