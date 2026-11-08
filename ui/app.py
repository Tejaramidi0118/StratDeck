"""
Phase 8.1: Playable StratDeck Card Game & AI Research Backend API.

Integrates:
- Real-time Playable Card Game Engine (Human Player 0 vs RL Opponent Player 1)
- Action Validation & Masking Enforcer
- Frozen MaskablePPO RL Agent (Player 1)
- Phase 4 Opponent Behavioral Profiler
- Phase 5 Profile-Aware DDA Controller
- Phase 6 Grounded LLM Explainer (Ollama qwen3.5:4b) & GroundingValidator
- Phase 7 Controlled Ablation Results & Visualizations

Strictly preserves frozen scientific artifacts (Phases 1–7).
"""

import sys
import os
import json
import uuid
import time
import urllib.request
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sb3_contrib import MaskablePPO
from env.gym_wrapper import AgentZeroEnv, NUM_ACTIONS
from env.cards import Card
from training.evaluate_ppo import load_any_model
from modules.explanation_evidence import extract_explanation_evidence, action_to_name
from modules.grounding_validator import GroundingValidator
from modules.explainer import ExplainerModule, OllamaBackend
from modules.profiler import OpponentProfiler
from modules.dda import DDAController, DEFAULT_DIFFICULTY_MAPPING

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load & Cache Frozen Artifacts at Startup
MODEL_PATH = "checkpoints/ppo/ppo_final.zip"
print("Initializing Playable StratDeck Engine...")
print(f"Loading frozen MaskablePPO model from {MODEL_PATH}...")
ppo_model = MaskablePPO.load(MODEL_PATH)

print("Loading opponent model cache for DDA...")
model_cache = {}
for lvl, info in DEFAULT_DIFFICULTY_MAPPING.items():
    if info["type"] == "model" and info["path"] and os.path.exists(info["path"]):
        model_cache[lvl] = load_any_model(info["path"])
    else:
        model_cache[lvl] = None

print("Loading Phase 4 Opponent Profiler...")
dataset_path = "data/trajectories_dataset.json"
profiler = OpponentProfiler(n_clusters=2, seed=42)
if os.path.exists(dataset_path):
    with open(dataset_path, "r") as f:
        ds = json.load(f)
    profiler.fit([d["move_history"] for d in ds])

# In-Memory Playable Game Sessions Storage
GAME_SESSIONS = {}


def serialize_ui_state(session_id: str, sess: dict) -> dict:
    """
    Serializes a UI-safe representation of the game state.
    STRICT COMPLIANCE: NEVER exposes opponent private hand or deck order.
    """
    env = sess["env"]
    controller = sess["controller"]
    engine = env.engine

    p0_obs = engine._get_obs(0)
    p0_hand = [action_to_name(int(c)) for c in p0_obs["hand"]]
    p0_mask = env.action_masks()

    public_discard = [action_to_name(int(c)) for c in engine.discard]

    p0_legal_action_ids = engine.legal_actions(0)
    p0_legal_names = [action_to_name(AgentZeroEnv._encode_action(a)) for a in p0_legal_action_ids]

    c_id = None
    if profiler is not None and len(sess["move_history"]) > 0:
        try:
            c_id = int(profiler.predict_cluster(sess["move_history"]))
        except Exception:
            c_id = None

    last_ai_ev = sess.get("last_ai_evidence")
    latest_ai_decision = None
    if last_ai_ev:
        pe = last_ai_ev["policy_evidence"]
        latest_ai_decision = {
            "action": pe["chosen_action_name"],
            "probability": pe["chosen_action_probability"],
            "rank": pe["chosen_action_rank"],
            "top_legal_actions": pe["top_legal_actions"],
            "value_estimate": pe["value_estimate_V_s"],
            "evidence": last_ai_ev,
            "explanation": sess.get("last_ai_explanation")
        }

    opp_hand_cnt = len(engine.players[1].hand) if len(engine.players) > 1 else 0
    p1_hand = None
    if sess.get("mode") == "watch" and len(engine.players) > 1:
        p1_hand = [action_to_name(int(c)) for c in engine.players[1].hand]

    deck_sz = p0_obs["deck_size"]
    hitman_danger = round((1.0 / max(1, deck_sz)) * 100, 1) if deck_sz > 0 else 0.0

    p0_label = "RL Agent 0 (MaskablePPO)" if sess.get("mode") == "watch" else "You (Player 0)"
    if sess.get("mode") == "practice":
        p1_label = "Tutorial Coach Bot (Rule-Based)"
    else:
        ckpt_name = controller.get_current_checkpoint_info()["name"]
        p1_label = f"RL Agent 1 ({ckpt_name})"

    payload = {
        "session_id": session_id,
        "turn_index": engine.turn_index,
        "current_player": engine.current_player,
        "game_over": engine.game_over,
        "winner": engine.winner,
        "deck_size": deck_sz,
        "hitman_danger_pct": hitman_danger,
        "pending_draw_count": p0_obs["pending_draw_count"],
        "alive_players_count": len(p0_obs["alive_players"]),
        "human_hand": p0_hand,
        "human_hand_counts": {c: p0_hand.count(c) for c in set(p0_hand)},
        "p0_hand": p0_hand,
        "opponent_hand_count": opp_hand_cnt,
        "public_discard": public_discard,
        "human_action_mask": p0_mask.tolist() if isinstance(p0_mask, np.ndarray) else p0_mask,
        "human_legal_actions": p0_legal_names,
        "action_log": sess["action_log"],
        "recent_ai_actions": sess.get("recent_ai_actions", []),
        "last_events": sess.get("last_events", []),
        "last_ai_summary": latest_ai_decision,
        "latest_ai_decision": latest_ai_decision,
        "dda_status": controller.get_status(),
        "cluster_id": c_id,
        "mode": sess.get("mode", "1v1"),
        "p0_name": p0_label,
        "p1_name": p1_label,
        "last_actor": sess.get("last_actor"),
        "last_actor_name": sess.get("last_actor_name"),
    }
    if p1_hand is not None:
        payload["p1_hand"] = p1_hand
    return payload


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    ollama_online = False
    models_available = []
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            ollama_online = True
            models_available = [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        ollama_online = False

    return jsonify({
        "status": "HEALTHY",
        "phase": "8.2",
        "mode": "Playable Card Game + AI Research Dashboard",
        "rl_model_loaded": ppo_model is not None,
        "model_cache_count": len(model_cache),
        "profiler_fitted": profiler.kmeans is not None,
        "ollama_online": ollama_online,
        "target_model": "qwen3.5:4b",
        "target_model_installed": any("qwen3.5:4b" in m for m in models_available)
    })


def create_ai_policy_wrapper(sess, controller):
    def policy_fn(raw_obs, legal_cards):
        env = sess["env"]
        obs_vec = AgentZeroEnv.encode_raw_obs(raw_obs)
        curr_lvl = controller.current_difficulty
        cached_model = model_cache.get(curr_lvl)

        ai_mask = np.zeros(NUM_ACTIONS, dtype=bool)
        for c in legal_cards:
            if c == env.engine.DRAW_ACTION or c == -1:
                ai_mask[9] = True
            else:
                ai_mask[int(c)] = True

        if curr_lvl == 0 or cached_model is None:
            # Level 0 (Random Agent): Uniform random selection among legal actions
            legal_choices = [9 if (c == env.engine.DRAW_ACTION or c == -1) else int(c) for c in legal_cards]
            ai_act_int = int(np.random.choice(legal_choices))
            eval_model = ppo_model  # Use ppo_model strictly for observational state/value estimation
        else:
            ai_act_idx, _ = cached_model.predict(obs_vec, action_masks=ai_mask, deterministic=True)
            ai_act_int = int(ai_act_idx)
            eval_model = cached_model

        if ai_act_int == 9:
            chosen_card = env.engine.DRAW_ACTION
        else:
            chosen_card = Card(ai_act_int)

        p1_history = list(env.engine.players[1].move_history)
        formatted_history = [[t, int(act) if act != -1 else -1] for t, act in p1_history]
        ai_evidence = extract_explanation_evidence(
            model=eval_model,
            env=env,
            obs=obs_vec,
            chosen_action_idx=ai_act_int,
            player_id=1,
            dda_controller=controller,
            profiler=profiler,
            move_history=formatted_history
        )
        sess["last_ai_evidence"] = ai_evidence
        sess["last_ai_explanation"] = None  # Reset explanation for new action

        ai_action_name = action_to_name(ai_act_int)
        sess["action_log"].append(f"Turn {env.engine.turn_index}: RL Agent played {ai_action_name}")

        # Accumulate multi-action turn sequence for visual card drop in UI
        sess.setdefault("recent_ai_actions", []).append({
            "action": ai_action_name,
            "probability": float(ai_evidence["policy_evidence"]["chosen_action_probability"]),
            "rank": int(ai_evidence["policy_evidence"]["chosen_action_rank"]),
            "top_legal_actions": ai_evidence["policy_evidence"]["top_legal_actions"],
            "value_estimate": float(ai_evidence["policy_evidence"]["value_estimate_V_s"]),
            "turn_index": int(env.engine.turn_index),
            "is_utility": False
        })

        return chosen_card

    return policy_fn


@app.route("/api/game/start", methods=["POST"])
def start_game():
    data = request.json or {}
    seed = int(data.get("seed", 42))
    mode = data.get("mode", "1v1")
    player_count = int(data.get("player_count", 2))
    difficulty = data.get("difficulty", "auto")

    session_id = str(uuid.uuid4())
    controller = DDAController(
        target_band=(0.45, 0.55),
        window_size=20,
        min_games_between_adjustments=5,
        initial_difficulty=2,
        profiler=profiler
    )

    if mode == "practice":
        controller.set_difficulty(1, fixed=True)
    elif difficulty != "auto" and difficulty is not None:
        try:
            diff_lvl = int(difficulty)
            controller.set_difficulty(diff_lvl, fixed=True)
        except (ValueError, TypeError):
            controller.is_fixed = False

    diff_desc = f"Fixed Tier {controller.current_difficulty} ({controller.get_current_checkpoint_info()['name']})" if controller.is_fixed else "Auto (Adaptive DDA)"

    is_watch = (mode == "watch")
    if is_watch:
        init_log = f"Watch Match Initialized (RL Agent 0 [MaskablePPO] vs RL Agent 1 [{diff_desc}]). Press Auto Play or Step Move to begin spectating."
    elif mode == "practice":
        init_log = "🎓 Tutorial Match Initialized. Use the Tutorial Coach box above your hand to learn what every card and move does step-by-step!"
    else:
        init_log = f"Game initialized ({mode.upper()} mode, Seed {seed}, Difficulty: {diff_desc}). You are Player 0."

    sess = {
        "env": None,
        "obs": None,
        "controller": controller,
        "seed": seed,
        "steps": 0,
        "move_history": [],
        "action_log": [init_log],
        "last_ai_evidence": None,
        "last_ai_explanation": None,
        "recent_ai_actions": [],
        "last_events": [],
        "mode": mode,
        "player_count": player_count,
        "last_actor": None,
        "last_actor_name": None
    }

    if is_watch:
        env = AgentZeroEnv(num_players=player_count, opponent_policy=None, seed=seed)
    else:
        ai_policy = create_ai_policy_wrapper(sess, controller)
        env = AgentZeroEnv(num_players=player_count, opponent_policy=ai_policy, seed=seed)

    sess["env"] = env

    obs, _ = env.reset(seed=seed)
    sess["obs"] = obs

    GAME_SESSIONS[session_id] = sess

    return jsonify(serialize_ui_state(session_id, sess))


@app.route("/api/game/watch/step", methods=["POST"])
def watch_step():
    data = request.json or {}
    session_id = data.get("session_id")
    if not session_id or session_id not in GAME_SESSIONS:
        return jsonify({"error": "Invalid or missing session_id"}), 400

    sess = GAME_SESSIONS[session_id]
    if sess.get("mode") != "watch":
        return jsonify({"error": "Session is not in watch mode"}), 400

    env = sess["env"]
    engine = env.engine
    controller = sess["controller"]

    if engine.game_over:
        return jsonify(serialize_ui_state(session_id, sess))

    actor_id = engine.current_player
    ckpt_info = controller.get_current_checkpoint_info()
    actor_label = "RL Agent 0 (MaskablePPO)" if actor_id == 0 else f"RL Agent 1 ({ckpt_info['name']})"

    if actor_id == 0:
        actor_model = ppo_model
    else:
        curr_lvl = controller.current_difficulty
        actor_model = model_cache.get(curr_lvl)

    legal_cards = engine.legal_actions(actor_id)
    ai_mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for c in legal_cards:
        if c == engine.DRAW_ACTION or c == -1:
            ai_mask[9] = True
        else:
            ai_mask[int(c)] = True

    raw_obs = engine._get_obs(actor_id)
    obs_vec = AgentZeroEnv.encode_raw_obs(raw_obs)

    if actor_id == 1 and (controller.current_difficulty == 0 or actor_model is None):
        legal_choices = [9 if (c == engine.DRAW_ACTION or c == -1) else int(c) for c in legal_cards]
        chosen_action_idx = int(np.random.choice(legal_choices))
        eval_model = ppo_model
    else:
        chosen_action_idx, _ = actor_model.predict(obs_vec, action_masks=ai_mask, deterministic=True)
        chosen_action_idx = int(chosen_action_idx)
        eval_model = actor_model

    if chosen_action_idx == 9:
        chosen_card = engine.DRAW_ACTION
    else:
        chosen_card = Card(chosen_action_idx)

    obs, reward, done, step_info = engine.step(chosen_card)

    p_history = list(engine.players[actor_id].move_history)
    formatted_history = [[t, int(act) if act != -1 else -1] for t, act in p_history]
    ai_evidence = extract_explanation_evidence(
        model=eval_model,
        env=env,
        obs=obs_vec,
        chosen_action_idx=chosen_action_idx,
        player_id=actor_id,
        dda_controller=controller,
        profiler=profiler,
        move_history=formatted_history
    )
    sess["last_ai_evidence"] = ai_evidence
    sess["last_ai_explanation"] = None

    action_name = action_to_name(chosen_action_idx)
    sess["action_log"].append(f"Turn {engine.turn_index}: {actor_label} played {action_name}")

    sess["recent_ai_actions"] = [{
        "actor_id": actor_id,
        "actor_label": actor_label,
        "action": action_name,
        "probability": float(ai_evidence["policy_evidence"]["chosen_action_probability"]),
        "rank": int(ai_evidence["policy_evidence"]["chosen_action_rank"]),
        "top_legal_actions": ai_evidence["policy_evidence"]["top_legal_actions"],
        "value_estimate": float(ai_evidence["policy_evidence"]["value_estimate_V_s"]),
        "turn_index": int(engine.turn_index),
        "is_utility": is_util
    }]

    sess["last_actor"] = actor_id
    sess["last_actor_name"] = actor_label

    last_events = []
    if action_name == "SPY":
        peek_cards = [action_to_name(int(c)) for c in engine.deck[-3:]]
        last_events.append({"type": "SPY_PEEK", "cards": peek_cards, "actor": actor_label})
    elif action_name == "TRAP":
        last_events.append({"type": "AI_TRAP", "message": f"{actor_label} armed a TRAP!"})
    sess["last_events"] = last_events

    if done:
        winner_name = "RL Agent 0 (MaskablePPO)" if engine.winner == 0 else f"RL Agent 1 ({ckpt_info['name']})"
        sess["action_log"].append(f"🏆 Match Concluded! Winner: {winner_name}")

    return jsonify(serialize_ui_state(session_id, sess))


@app.route("/api/game/difficulty", methods=["POST"])
def set_game_difficulty():
    data = request.json or {}
    session_id = data.get("session_id")
    if not session_id or session_id not in GAME_SESSIONS:
        return jsonify({"error": "Invalid or missing session_id"}), 400

    sess = GAME_SESSIONS[session_id]
    controller = sess["controller"]
    difficulty = data.get("difficulty", "auto")

    if difficulty == "auto" or difficulty is None:
        controller.is_fixed = False
        sess["action_log"].append("Difficulty switched to: Auto (DDA Adaptive Controller)")
    else:
        try:
            lvl = int(difficulty)
            controller.set_difficulty(lvl, fixed=True)
            ckpt_info = controller.get_current_checkpoint_info()
            sess["action_log"].append(f"Difficulty switched to: Level {lvl} ({ckpt_info['name']})")
        except (ValueError, TypeError):
            return jsonify({"error": f"Invalid difficulty value: {difficulty}"}), 400

    return jsonify(serialize_ui_state(session_id, sess))


@app.route("/api/game/state", methods=["GET"])
def get_game_state():
    session_id = request.args.get("session_id")
    if not session_id or session_id not in GAME_SESSIONS:
        return jsonify({"error": "Invalid or missing session_id"}), 400
    sess = GAME_SESSIONS[session_id]
    return jsonify(serialize_ui_state(session_id, sess))


@app.route("/api/game/action", methods=["POST"])
def submit_human_action():
    data = request.json or {}
    session_id = data.get("session_id")
    action_idx = data.get("action_idx")

    if not session_id or session_id not in GAME_SESSIONS:
        return jsonify({"error": "Invalid or expired session ID"}), 400

    sess = GAME_SESSIONS[session_id]
    if sess.get("mode") == "watch":
        return jsonify({"error": "Cannot submit human actions in Watch Match spectator mode. Use /api/game/watch/step to step through RL agent moves."}), 400

    env = sess["env"]
    controller = sess["controller"]

    if env.engine.game_over:
        return jsonify({"error": "Game is already completed."}), 400

    if env.engine.current_player != 0:
        return jsonify({"error": "Not your turn! It is currently Player " + str(env.engine.current_player) + "'s turn."}), 400

    # 1. Validate Human Action Mask
    mask = env.action_masks()
    if action_idx is None or action_idx < 0 or action_idx >= NUM_ACTIONS or not mask[action_idx]:
        return jsonify({"error": f"Action index {action_idx} is currently masked/illegal for Human player."}), 400

    human_action_name = action_to_name(action_idx)
    sess["action_log"].append(f"Turn {env.engine.turn_index}: You played {human_action_name}")

    # Reset per-round recent lists before executing
    sess["recent_ai_actions"] = []
    sess["last_events"] = []

    human_hand_before = list(env.engine.players[0].hand)

    # 2. Execute Human Action (and automatic AI turn loop via AgentZeroEnv.opponent_policy)
    obs, reward, done, truncated, info = env.step(action_idx)
    sess["obs"] = obs
    sess["steps"] += 1

    # Record Human Action Events for Visual UI Popups / Reveal Trays
    if human_action_name == "SPY" and "spy_peek" in info:
        peek_names = [action_to_name(int(c)) for c in info["spy_peek"]]
        sess["last_events"].append({
            "type": "SPY_PEEK",
            "actor": "human",
            "cards": peek_names,
            "message": f"👁️ SPY REVEAL: Top {len(peek_names)} cards of draw deck peeked!"
        })
    elif human_action_name == "STEAL" and "stolen_card" in info and info["stolen_card"] is not None:
        stolen_name = action_to_name(int(info["stolen_card"]))
        sess["last_events"].append({
            "type": "STEAL_SUCCESS",
            "actor": "human",
            "card": stolen_name,
            "message": f"💰 YOU STOLE {stolen_name} from RL Agent!"
        })
    elif human_action_name == "SWAP" and "swap_taken" in info and info["swap_taken"] is not None:
        given_name = action_to_name(int(info["swap_given"])) if info["swap_given"] is not None else "None"
        taken_name = action_to_name(int(info["swap_taken"]))
        sess["last_events"].append({
            "type": "SWAP_SUCCESS",
            "actor": "human",
            "given": given_name,
            "taken": taken_name,
            "message": f"🔀 CARD SWAP: You gave {given_name} and received {taken_name}!"
        })

    # Record AI Events for UI Popups (e.g. if AI stole a card from human)
    opp_infos = info.get("opponent_infos", [])
    stolen_cards_list = []
    for idx, ai_act in enumerate(sess.get("recent_ai_actions", [])):
        opp_info = opp_infos[idx] if idx < len(opp_infos) else {}
        if ai_act["action"] == "STEAL":
            if "stolen_card" in opp_info and opp_info["stolen_card"] is not None:
                stolen_card = action_to_name(int(opp_info["stolen_card"]))
                stolen_cards_list.append(stolen_card)
                sess["last_events"].append({
                    "type": "AI_STOLE",
                    "actor": "ai",
                    "card": stolen_card,
                    "message": f"⚠️ RL AGENT STOLE {stolen_card} from your hand!"
                })
            else:
                sess["last_events"].append({
                    "type": "AI_STOLE_FAILED",
                    "actor": "ai",
                    "message": "⚠️ RL Agent tried to STEAL, but your hand was empty!"
                })
        elif ai_act["action"] == "SWAP":
            sess["last_events"].append({
                "type": "SWAP_SUCCESS",
                "actor": "ai",
                "message": "🔀 RL AGENT SWAPPED CARDS: Exchanged a random card with your hand!"
            })
        elif ai_act["action"] == "TRAP":
            sess["last_events"].append({
                "type": "AI_TRAP",
                "actor": "ai",
                "message": "🪤 RL AGENT SET A TRAP: You must draw 2 cards on your next draw!"
            })
        elif ai_act["action"] == "ATTACK":
            sess["last_events"].append({
                "type": "AI_ATTACK",
                "actor": "ai",
                "message": "⚔️ RL AGENT PLAYED ATTACK: Extra draw stack passed to you!"
            })

    if stolen_cards_list:
        summary_stolen = ", ".join(stolen_cards_list)
        sess["action_log"].append(f"⚠️ HAND THEFT: RL Agent executed STEAL and took {len(stolen_cards_list)} card(s) from your hand: {summary_stolen}")

    if info.get("defused") or any(o.get("defused") for o in opp_infos):
        sess["last_events"].append({
            "type": "DEFUSED",
            "message": "🛡️ HITMAN DEFUSED BY SHIELD! Card shuffled back into draw deck!"
        })

    p0_history = list(env.engine.players[0].move_history)
    formatted_history = [[t, int(act) if act != -1 else -1] for t, act in p0_history]
    sess["move_history"] = formatted_history

    if done:
        p0_won = (env.engine.winner == 0)
        controller.update(player_won=p0_won, move_history=formatted_history, game_length=sess["steps"])
        sess["action_log"].append(f"Game Ended! {'YOU WIN!' if p0_won else 'RL AGENT WINS!'}")

    return jsonify(serialize_ui_state(session_id, sess))


@app.route("/api/explain", methods=["POST"])
def explain_decision():
    data = request.json or {}
    session_id = data.get("session_id")

    if not session_id or session_id not in GAME_SESSIONS:
        return jsonify({"error": "Invalid session ID"}), 400

    sess = GAME_SESSIONS[session_id]
    evidence = sess.get("last_ai_evidence")

    if not evidence:
        return jsonify({"error": "No RL Agent decision recorded yet. Play a turn first."}), 400

    t0 = time.time()
    used_model = "qwen3.5:4b"
    try:
        backend = OllamaBackend(model="qwen3.5:4b")
        explainer = ExplainerModule(backend=backend)
        exp_result = explainer.explain_move(evidence)
    except Exception as e:
        print(f"Ollama local inference unavailable ({e}), using evidence-grounded template fallback.")
        from modules.explainer import MockBackend
        fallback_explainer = ExplainerModule(backend=MockBackend())
        exp_result = fallback_explainer.explain_move(evidence)
        used_model = "Evidence-Grounded Template (Offline Mode)"

    elapsed = time.time() - t0
    sess["last_ai_explanation"] = exp_result
    
    validator = GroundingValidator()
    val_result = validator.validate(evidence, exp_result)
    val_result["passed"] = bool(val_result.get("grounded", False))

    return jsonify({
        "status": "SUCCESS",
        "model": used_model,
        "elapsed_seconds": round(elapsed, 2),
        "ai_action": evidence["policy_evidence"]["chosen_action_name"],
        "explanation": exp_result.get("explanation", ""),
        "bullet_evidence": exp_result.get("evidence", []),
        "parsed_json": exp_result.get("grounding_status") != "UNPARSED",
        "validation": val_result
    })


@app.route("/api/experiments", methods=["GET"])
def get_experiments():
    ablation_file = "data/ablation/results/ablation_summary_metrics.json"
    phase3_file = "data/results.json"

    ablation_data = {}
    if os.path.exists(ablation_file):
        with open(ablation_file, "r") as f:
            ablation_data = json.load(f)

    phase3_data = {}
    if os.path.exists(phase3_file):
        with open(phase3_file, "r") as f:
            phase3_data = json.load(f)

    return jsonify({
        "ablation_studies": ablation_data,
        "phase3_comparison": phase3_data,
    })


@app.route("/plots/<path:filename>")
def serve_plot(filename):
    plot_dir = os.path.abspath("data/ablation/plots")
    return send_from_directory(plot_dir, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
