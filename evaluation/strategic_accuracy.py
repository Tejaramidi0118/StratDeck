"""
Strategic accuracy, decision quality and calibration for the StratDeck agent.

Produces the metrics defined in Section "Strategic Accuracy, Decision Quality
and Calibration" of the report:

  * Acc@1 / Acc@3   agreement with a determinized Monte-Carlo reference
  * mean decision loss and blunder rate (win-probability loss per decision)
  * Brier score, skill score, ECE, ROC-AUC, reliability bins for the critic
  * Spearman correlation between true hazard and DRAW propensity

Run from the project root:
    python evaluation/strategic_accuracy.py --episodes 150 --rollouts 32
Writes data/strategic_accuracy.json
"""

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch

from env.game_engine import GameEngine
from env.cards import Card
from env.gym_wrapper import AgentZeroEnv, NUM_ACTIONS
from sb3_contrib import MaskablePPO

DRAW_IDX = 9


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def legal_mask(engine, pid):
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for a in engine.legal_actions(pid):
        mask[DRAW_IDX if a == GameEngine.DRAW_ACTION else int(a)] = True
    return mask


def idx_to_action(i):
    return GameEngine.DRAW_ACTION if i == DRAW_IDX else Card(i)


def clone_engine(engine):
    """Deep-ish copy sufficient to roll a determinization forward."""
    import copy
    return copy.deepcopy(engine)


def true_hazard(engine, pid):
    """P(next single draw is lethal), computable from public info by elimination."""
    deck = len(engine.deck)
    if deck == 0:
        return 0.0
    hitmen = sum(1 for c in engine.deck if c == Card.HITMAN)
    return hitmen / deck


# --------------------------------------------------------------------------
# determinized Monte-Carlo reference
# --------------------------------------------------------------------------
def determinized_action_values(engine, pid, n_rollouts, rng, horizon=120):
    """
    Sample `n_rollouts` consistent continuations per legal action and return
    {action_index: estimated win probability for `pid`}.

    Determinization note: the opponent hand and deck order are resampled from
    the cards not publicly accounted for. This ignores the value of hiding
    information, so SPY-type actions are systematically undervalued. It is a
    reference, not an oracle -- the report says so explicitly.
    """
    values = {}
    for a_idx in np.flatnonzero(legal_mask(engine, pid)):
        wins = 0
        for _ in range(n_rollouts):
            sim = clone_engine(engine)
            sim.rng = random.Random(rng.randrange(1 << 30))
            # shuffle the unseen portion: deck order is not known to the player
            sim.rng.shuffle(sim.deck)
            try:
                sim.step(idx_to_action(int(a_idx)))
            except Exception:
                continue
            steps = 0
            while not sim.game_over and steps < horizon:
                cur = sim.current_player
                la = sim.legal_actions(cur)
                try:
                    sim.step(sim.rng.choice(la))
                except Exception:
                    break
                steps += 1
            if sim.winner == pid:
                wins += 1
        values[int(a_idx)] = wins / max(1, n_rollouts)
    return values


# --------------------------------------------------------------------------
# main evaluation
# --------------------------------------------------------------------------
def evaluate(model_path, episodes, n_rollouts, blunder_tau, seed, max_steps=300):
    rng = random.Random(seed)
    np.random.seed(seed)
    model = MaskablePPO.load(model_path, device="cpu")

    agree1 = agree3 = n_dec = 0
    losses = []
    blunders = 0
    hazard_pairs = []              # (true hazard, chose DRAW)
    calib = []                     # (predicted win prob, outcome)

    for ep in range(episodes):
        engine = GameEngine(num_players=2, seed=50_000 + ep)
        engine.reset()
        ep_records = []

        for _ in range(max_steps):
            if engine.game_over:
                break
            pid = engine.current_player
            raw = engine._get_obs(pid)
            obs = AgentZeroEnv.encode_raw_obs(raw)
            mask = legal_mask(engine, pid)

            if pid == 0:
                act, _ = model.predict(obs, action_masks=mask, deterministic=True)
                act = int(act)

                # critic value -> win probability (logistic squash of the value head)
                with torch.no_grad():
                    t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    v = float(model.policy.predict_values(t).squeeze())
                ep_records.append((1.0 / (1.0 + np.exp(-v / 50.0)),))

                hazard_pairs.append((true_hazard(engine, pid), 1 if act == DRAW_IDX else 0))

                if n_rollouts > 0:
                    qref = determinized_action_values(engine, pid, n_rollouts, rng)
                    if qref:
                        ranked = sorted(qref.items(), key=lambda kv: -kv[1])
                        best_val = ranked[0][1]
                        agree1 += int(ranked[0][0] == act)
                        agree3 += int(act in [k for k, _ in ranked[:3]])
                        loss = best_val - qref.get(act, 0.0)
                        losses.append(loss)
                        blunders += int(loss > blunder_tau)
                        n_dec += 1
            else:
                act = int(rng.choice(np.flatnonzero(mask)))

            try:
                engine.step(idx_to_action(act))
            except Exception:
                break

        outcome = 1 if engine.winner == 0 else 0
        for (p,) in ep_records:
            calib.append((p, outcome))

    return _summarize(agree1, agree3, n_dec, losses, blunders,
                      hazard_pairs, calib, blunder_tau)


# --------------------------------------------------------------------------
# metric computation
# --------------------------------------------------------------------------
def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def _auc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, int)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, int)
    edges = np.linspace(0, 1, bins + 1)
    total, rows = 0.0, []
    for i in range(bins):
        sel = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if sel.sum() == 0:
            continue
        acc, conf = float(y[sel].mean()), float(p[sel].mean())
        total += sel.sum() / len(p) * abs(acc - conf)
        rows.append({"bin": [round(edges[i], 2), round(edges[i + 1], 2)],
                     "n": int(sel.sum()), "accuracy": round(acc, 4),
                     "confidence": round(conf, 4)})
    return float(total), rows


def _summarize(agree1, agree3, n_dec, losses, blunders, hazard_pairs, calib, tau):
    out = {"n_reference_decisions": n_dec, "blunder_threshold": tau}

    if n_dec:
        out["agreement"] = {
            "acc_at_1": round(agree1 / n_dec, 4),
            "acc_at_3": round(agree3 / n_dec, 4),
        }
        out["decision_quality"] = {
            "mean_decision_loss": round(float(np.mean(losses)), 4),
            "median_decision_loss": round(float(np.median(losses)), 4),
            "blunder_rate": round(blunders / n_dec, 4),
        }

    if calib:
        p = np.array([c[0] for c in calib])
        y = np.array([c[1] for c in calib])
        brier = float(np.mean((p - y) ** 2))
        base = float(y.mean() * (1 - y.mean()))
        ece, rows = _ece(p, y)
        out["calibration"] = {
            "n": len(p),
            "brier_score": round(brier, 4),
            "reference_brier": round(base, 4),
            "skill_score": round(1 - brier / base, 4) if base > 0 else None,
            "ece": round(ece, 4),
            "roc_auc": round(_auc(p, y), 4),
            "reliability_bins": rows,
        }

    if hazard_pairs:
        h = [a for a, _ in hazard_pairs]
        d = [b for _, b in hazard_pairs]
        binned = defaultdict(list)
        for hv, dv in hazard_pairs:
            binned[round(min(hv, 0.49) * 10) / 10].append(dv)
        out["hazard_response"] = {
            "spearman_hazard_vs_draw": round(_spearman(h, d), 4),
            "draw_rate_by_hazard_bin": {str(k): round(float(np.mean(v)), 4)
                                        for k, v in sorted(binned.items())},
            "note": "Spearman should be strongly negative: a competent policy "
                    "draws less as the lethal-draw probability rises.",
        }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/ppo/ppo_final.zip")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--rollouts", type=int, default=32,
                    help="determinizations per action; 0 skips the reference metrics")
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/strategic_accuracy.json")
    a = ap.parse_args()

    res = evaluate(a.model, a.episodes, a.rollouts, a.tau, a.seed)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print(f"\n[saved] {a.out}")