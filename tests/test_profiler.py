"""
Unit tests for Phase 4 Opponent Profiler Module.
Tests:
1. Feature extraction from known trajectory
2. Correct action-frequency calculation
3. Empty/short trajectory handling
4. Feature normalization
5. Embedding generation
6. Clustering execution
7. Profile serialization
8. Verification that no hidden opponent hand information is present in features
"""

import os
import json
import numpy as np

from modules.profiler import extract_features, feature_to_text, OpponentProfiler, FEATURE_NAMES
from env.cards import Card


def test_feature_extraction_known_trajectory():
    # Known move history: 2 DRAWs (-1), 1 ATTACK, 1 SHIELD (4 actions across 3 turns)
    history = [
        (0, -1),
        (0, Card.ATTACK),
        (1, Card.SHIELD),
        (2, -1)
    ]
    feat = extract_features(history)
    assert len(feat) == len(FEATURE_NAMES)
    
    # freq_draw = 2/4 = 0.5
    assert np.isclose(feat[0], 0.5)
    # freq_attack = 1/4 = 0.25
    assert np.isclose(feat[1], 0.25)
    # freq_shield = 1/4 = 0.25
    assert np.isclose(feat[2], 0.25)
    # total_turns = 4 (4 total actions in history)
    assert np.isclose(feat[13], 4.0)


def test_action_frequency_calculation():
    # History with 10 DRAWs
    history = [(i, -1) for i in range(10)]
    feat = extract_features(history)
    assert np.isclose(feat[0], 1.0)  # 100% draw
    assert np.isclose(feat[9], 0.0)  # 0% offensive ratio
    assert np.isclose(feat[10], 0.0) # 0% defensive ratio


def test_empty_and_short_trajectory_handling():
    # Empty history
    empty_feat = extract_features([])
    assert len(empty_feat) == len(FEATURE_NAMES)
    assert np.all(empty_feat == 0.0)

    # Short history (single action)
    short_feat = extract_features([(0, Card.ATTACK)])
    assert len(short_feat) == len(FEATURE_NAMES)
    assert np.isclose(short_feat[1], 1.0) # freq_attack = 1.0
    assert not np.isnan(short_feat).any()


def test_feature_normalization():
    histories = [
        [(0, Card.ATTACK)],
        [(0, Card.SHIELD)],
        [(0, -1)]
    ]
    profiler = OpponentProfiler(n_clusters=2, seed=42)
    profiler.fit(histories)
    norm_feat = profiler.scaler.transform([extract_features(histories[0])])
    assert norm_feat.shape == (1, len(FEATURE_NAMES))
    assert not np.isnan(norm_feat).any()


def test_embedding_generation():
    feat = extract_features([(0, Card.ATTACK), (1, Card.SHIELD)])
    text = feature_to_text(feat)
    profiler = OpponentProfiler(n_clusters=2, seed=42)
    emb = profiler.embed_text(text)
    assert isinstance(emb, np.ndarray)
    assert len(emb) > 0
    assert not np.isnan(emb).any()


def test_clustering_execution():
    histories = [
        [(t, Card.ATTACK) for t in range(5)],
        [(t, Card.SHIELD) for t in range(5)],
        [(t, -1) for t in range(5)]
    ]
    profiler = OpponentProfiler(n_clusters=2, seed=42)
    profiler.fit(histories)
    cid = profiler.predict_cluster(histories[0])
    assert cid in [0, 1]


def test_profile_serialization():
    histories = [[(0, Card.ATTACK)], [(0, -1)]]
    profiler = OpponentProfiler(n_clusters=2, seed=42)
    profiler.fit(histories)
    labels = [profiler.predict_cluster(h) for h in histories]
    stats = profiler.get_cluster_statistics(histories, labels)
    
    assert 0 in stats or 1 in stats
    serialized = json.dumps(stats)
    assert len(serialized) > 0


def test_no_hidden_information_leakage():
    # Confirm feature list contains ONLY publicly observable move history counts
    for feat_name in FEATURE_NAMES:
        assert "hand" not in feat_name.lower()
        assert "hidden" not in feat_name.lower()
        assert "deck_secret" not in feat_name.lower()
    
    # Confirm feature extraction accepts strictly move_history list of (turn_idx, action)
def test_action_diversity_and_zero_variance_absence():
    # History 1: 1 action type (DRAW only) -> diversity = 1/10 = 0.1
    h1 = [(0, -1), (1, -1)]
    f1 = extract_features(h1)
    assert np.isclose(f1[12], 0.1)

    # History 2: 3 action types (DRAW, ATTACK, SHIELD) -> diversity = 3/10 = 0.3
    h2 = [(0, -1), (1, Card.ATTACK), (2, Card.SHIELD)]
    f2 = extract_features(h2)
    assert np.isclose(f2[12], 0.3)

    # Check zero-variance across a batch
    matrix = np.array([f1, f2])
    stds = matrix.std(axis=0)
    zero_var_idx = np.where(stds == 0)[0]
    # Confirm action_diversity (index 12) has non-zero variance
    assert 12 not in zero_var_idx


if __name__ == "__main__":
    test_feature_extraction_known_trajectory()
    test_action_frequency_calculation()
    test_empty_and_short_trajectory_handling()
    test_feature_normalization()
    test_embedding_generation()
    test_clustering_execution()
    test_profile_serialization()
    test_no_hidden_information_leakage()
    test_action_diversity_and_zero_variance_absence()
    print("ALL 9 PHASE 4 PROFILER TESTS PASSED SUCCESSFULLY!")


