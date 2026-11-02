# Agent Zero — Setup & Run Order

## 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate         # on Mac/Linux
pip install -r requirements.txt
```

If using Ollama for the explainer module (Phase 6):
```bash
brew install ollama              # Mac
ollama pull llama3.1
ollama serve                     # keep this running in a separate terminal
```

## 2. Run order (matches the 8-phase plan)

**Phase 1 — validate the game engine:**
```bash
python test_phase1.py
```
Expect roughly balanced win rates (~45-55% each) between two random bots, few timeouts.

**Phase 2 — train the base self-play PPO agent:**
```bash
python -m training.train_ppo
```
Success criterion: win-rate vs random baseline > 70%. This will take a while;
reduce `TOTAL_TIMESTEPS` in the script for a quick smoke-test first.

**Phase 3 — train comparison algorithms and compare:**
```bash
python -m training.train_dqn
python -m training.train_a2c
python -m training.compare_algorithms
```

**Phase 4 — opponent profiling (offline test, no training needed):**
```python
from modules.profiler import OpponentProfiler
# see modules/profiler.py docstring / evaluation/conflict_eval.py for usage
```

**Phase 5 — DDA (unit-testable standalone):**
```python
from modules.dda import DifficultyAdjuster
# see modules/dda.py - call record_result() after each game, check .status()
```

**Phase 6 — LLM explainer (requires Ollama running, or a Groq API key):**
```python
from modules.explainer import ExplainerModule, OllamaBackend
explainer = ExplainerModule(backend=OllamaBackend(model="llama3.1"))
# see bottom of modules/explainer.py for a full usage example
```

**Phase 7 — the core research measurements:**
```bash
python -m evaluation.faithfulness_eval
python -m evaluation.conflict_eval
```
Record the printed numbers into `data/results.json` (dashboard.py will scaffold
this file for you on first run if it doesn't exist).

**Phase 8 — UI and dashboard:**
```bash
python -m ui.pygame_app       # human vs trained AI, with live explanations
python -m ui.dashboard        # generates data/plot_*.png from data/results.json
```

## 3. Swapping the LLM backend later

Only `modules/explainer.py`'s backend argument needs to change:
```python
# Local (default, free, for development):
explainer = ExplainerModule(backend=OllamaBackend(model="llama3.1"))

# Hosted (swap in later for quality/speed):
explainer = ExplainerModule(backend=GroqBackend(api_key="YOUR_GROQ_KEY"))
```
No other file needs to change.

## 4. Known simplifications (be ready to explain these in the viva)

- The self-play checkpoint pool (`training/self_play_pool.py`) uses a simple
  70/30 recent-vs-random-older sampling heuristic — reasonable for a course
  project, but a documented simplification vs. more advanced self-play methods.
- `evaluation/faithfulness_eval.py`'s faithfulness check (does the explanation
  text mention the same card name as the real top action) is intentionally
  simple. Consider a stricter semantic-similarity check if time allows.
- The Pygame UI's action-probability display uses a placeholder; wire in the
  real `model.policy.get_distribution(obs).distribution.probs` for the actual
  demo so the explainer is grounded in real numbers, not the placeholder shown
  in the current UI code.
