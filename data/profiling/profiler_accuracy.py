"""
Supervised accuracy evaluation for the StratDeck opponent profiler.

The profiler is the one component to which ordinary classification accuracy
genuinely applies: scripted players can be generated with KNOWN archetype
labels, so the recovered clusters can be scored against ground truth.

Reports Adjusted Rand Index, Hungarian-matched purity, per-class precision/
recall/F1, a confusion matrix, and a silhouette-based check on the choice of K.

Run from the project root:
    python evaluation/profiler_accuracy.py --matches-per-archetype 60
Writes data/profiler_accuracy.json
"""

import argparse
import json
import os
import random
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.optimize import linear_sum_assignment

from env.game_engine import GameEngine
from env.cards import Card

DRAW = GameEngine.DRAW_ACTION

# Archetypes are defined by a preference ordering over cards. These are the
# ground-truth labels the profiler must recover from public play alone.
ARCHETYPES = {
    "aggressive": [Card.ATTACK, Card.TRAP, Card.STEAL],
    "defensive":  [Card.SHIELD, Card.SKIP, Card.REVERSE],
    "disruptive": [Card.SPY, Card.SWAP, Card.STEAL],
    "hoarder":    [],   # strongly prefers DRAW
}


def scripted_player(archetype, rng, p_follow=0.80):
    """Returns a policy fn(legal_actions) -> action with the given bias."""
    prefs = ARCHETYPES[archetype]

    def policy(legal):
        if archetype == "hoarder":
            if DRAW in legal and rng.random() < p_follow:
                return DRAW
            return rng.choice(legal)
        if rng.random() < p_follow:
            for c in prefs:
                if c in legal:
                    return c
        return rng.choice(legal)

    return policy


def play_and_log(archetype, seed, max_steps=300):
    """Play one match; return the archetype player's PUBLIC action log."""
    rng = random.Random(seed)
    engine = GameEngine(num_players=2, seed=seed)
    engine.reset()
    target = scripted_player(archetype, rng)
    other = scripted_player("hoarder", random.Random(seed + 7))

    log, turns = [], 0
    for _ in range(max_steps):
        if engine.game_over:
            break
        pid = engine.current_player
        legal = engine.legal_actions(pid)
        act = target(legal) if pid == 0 else other(legal)
        if pid == 0:
            log.append(act)          # only what an opponent could observe
        try:
            engine.step(act)
        except Exception:
            break
        turns += 1
    return log, turns


def descriptor(log, turns):
    """The 14-dimensional behavioural descriptor used by the profiler."""
    n = max(1, len(log))
    cnt = Counter(log)
    order = [Card.HITMAN, Card.ATTACK, Card.SHIELD, Card.SKIP, Card.REVERSE,
             Card.SPY, Card.SWAP, Card.TRAP, Card.STEAL]
    f = [cnt.get(c, 0) / n for c in order]
    f_draw = cnt.get(DRAW, 0) / n
    freqs = [f_draw] + f[1:]                              # 9 frequencies

    off = (cnt.get(Card.ATTACK, 0) + cnt.get(Card.TRAP, 0) + cnt.get(Card.STEAL, 0)) / n
    deff = (cnt.get(Card.SHIELD, 0) + cnt.get(Card.SKIP, 0) + cnt.get(Card.REVERSE, 0)) / n
    util = (cnt.get(Card.SPY, 0) + cnt.get(Card.SWAP, 0)) / n

    probs = [v / n for v in cnt.values() if v > 0]
    diversity = -sum(p * np.log(p) for p in probs) if probs else 0.0

    return np.array(freqs + [off, deff, util, diversity, float(turns)], dtype=float)


def hungarian_match(y_true, y_pred, labels):
    """Map cluster ids to archetype labels by maximum-overlap assignment."""
    k = len(labels)
    cost = np.zeros((k, k))
    for i, lab in enumerate(labels):
        for j in range(k):
            cost[i, j] = -np.sum((y_true == i) & (y_pred == j))
    r, c = linear_sum_assignment(cost)
    return {int(cj): int(ri) for ri, cj in zip(r, c)}


def main(matches, k, seed, out):
    labels = list(ARCHETYPES.keys())
    X, y = [], []
    for li, arch in enumerate(labels):
        for m in range(matches):
            log, turns = play_and_log(arch, seed=seed + li * 10_000 + m)
            if len(log) >= 4:                # need some history to profile
                X.append(descriptor(log, turns))
                y.append(li)
    X, y = np.array(X), np.array(y)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Xs)
    pred = km.labels_

    mapping = hungarian_match(y, pred, labels)
    mapped = np.array([mapping.get(int(p), -1) for p in pred])

    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y, mapped):
        if p >= 0:
            cm[t, p] += 1

    per_class = {}
    for i, lab in enumerate(labels):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[lab] = {"precision": round(prec, 4),
                          "recall": round(rec, 4),
                          "f1": round(f1, 4),
                          "support": int(cm[i, :].sum())}

    # does the chosen K actually fit the data?
    sweep = {}
    for kk in range(2, min(7, len(Xs))):
        try:
            lbl = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit_predict(Xs)
            sweep[kk] = round(float(silhouette_score(Xs, lbl)), 4)
        except Exception:
            pass

    res = {
        "n_samples": int(len(y)),
        "n_archetypes": len(labels),
        "k_used": k,
        "adjusted_rand_index": round(float(adjusted_rand_score(y, pred)), 4),
        "purity": round(float(np.trace(cm) / cm.sum()), 4) if cm.sum() else None,
        "macro_f1": round(float(np.mean([v["f1"] for v in per_class.values()])), 4),
        "per_class": per_class,
        "confusion_matrix": {"rows_true": labels, "cols_pred": labels,
                             "matrix": cm.tolist()},
        "silhouette_by_k": sweep,
        "note": "ARI = 0 means chance-level agreement, 1 a perfect partition. "
                "Features are standardized before clustering; without this the "
                "turn-count feature dominates the Euclidean objective.",
    }

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches-per-archetype", type=int, default=60)
    ap.add_argument("--k", type=int, default=4, help="set to match the profiler's K")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/profiler_accuracy.json")
    a = ap.parse_args()
    main(a.matches_per_archetype, a.k, a.seed, a.out)