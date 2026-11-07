"""
Phase 6: Grounded LLM Explanation System Module.

Generates natural-language action justifications grounded strictly in observable
game-state evidence and real PPO policy inference probabilities pi(a|s).

Key Constraints:
1. LLM NEVER selects or modifies actions.
2. Explanations use strictly observable state facts and real PPO probabilities/values.
3. Zero Q-values claimed for PPO.
4. Zero hidden opponent information exposed.
5. Swappable backend: MockBackend (offline/testing), GroqBackend (hosted API), OllamaBackend (local API).
"""

import sys
import os
import json
import time
import urllib.request
from abc import ABC, abstractmethod

from modules.grounding_validator import GroundingValidator


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...


class MockBackend(LLMBackend):
    """Deterministic offline backend for testing and evidence-grounded demonstration."""

    def generate(self, prompt: str) -> str:
        try:
            lines = prompt.splitlines()
            chosen_action = "DRAW"
            chosen_prob = 0.50
            alternatives = []
            v_s = 0.0

            deck_sz = 20
            for line in lines:
                if "- Selected Action:" in line:
                    chosen_action = line.split("- Selected Action:")[1].strip().split()[0]
                elif "- Selected Action Probability:" in line:
                    p_str = line.split("- Selected Action Probability:")[1].strip().replace("%", "")
                    chosen_prob = float(p_str) / 100.0 if float(p_str) > 1.0 else float(p_str)
                elif "- Value Estimate V(s):" in line:
                    v_s = float(line.split("- Value Estimate V(s):")[1].strip())
                elif "- Deck Size:" in line:
                    try:
                        deck_sz = int(line.split("- Deck Size:")[1].strip().split()[0])
                    except Exception:
                        pass
                elif "- Available Legal Actions & Policy Probabilities:" in line or "- Top Legal Actions:" in line:
                    raw_alts = line.split(":")[-1].strip()
                    parts = raw_alts.split(",")
                    for part in parts:
                        p = part.strip()
                        if "(" in p and ")" in p:
                            act_name = p.split("(")[0].strip()
                            prob_str = p.split("(")[1].replace("%)", "").replace("%", "").strip()
                            prob_val = float(prob_str) / 100.0
                            if act_name.upper() != chosen_action.upper():
                                alternatives.append({"action": act_name, "probability": prob_val})

            alt_str = ", ".join([f"{a['action']} ({a['probability']:.1%})" for a in alternatives[:2]]) if alternatives else "none"

            res = {
                "selected_action": chosen_action,
                "explanation": (
                    f"The MaskablePPO agent selected {chosen_action} with a policy probability of {chosen_prob:.1%}, "
                    f"which was the top-ranked move among legal options. With a deck size of {deck_sz} cards, "
                    f"this choice strategically balances survival risk against legal alternatives ({alt_str}). "
                    f"The model's state-value estimate V(s) is {v_s:.2f}, reflecting expected long-term match return."
                ),
                "evidence": [
                    f"Selected action {chosen_action} has policy probability {chosen_prob:.1%}",
                    f"Legal alternatives evaluated: {alt_str}",
                    f"Deck size is {deck_sz} cards with state value V(s) = {v_s:.2f}"
                ],
                "legal_alternatives": alternatives[:3],
                "grounding_status": "SUPPORTED"
            }
            return json.dumps(res)
        except Exception:
            return json.dumps({
                "selected_action": "DRAW",
                "explanation": "The RL agent selected DRAW based on available policy probabilities.",
                "evidence": ["Policy selected DRAW"],
                "legal_alternatives": [],
                "grounding_status": "SUPPORTED"
            })


class GroqBackend(LLMBackend):
    """Hosted backend via Groq OpenAI-compatible API."""

    def __init__(self, api_key: str = None, model: str = "llama-3.1-70b-versatile"):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")

        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"].strip()


class OllamaBackend(LLMBackend):
    """Local backend via Ollama."""

    def __init__(self, model: str = "qwen3.5:4b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host

    def generate(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "think": False,
            "options": {
                "num_predict": 500,
                "temperature": 0.2
            }
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        
        last_exc = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                text = result.get("response", "").strip()
                if not text and "thinking" in result:
                    text = result.get("thinking", "").strip()
                if "<think>" in text:
                    text = text.split("</think>")[-1].strip()
                if text:
                    return text
            except Exception as e:
                last_exc = e
                time.sleep(1)
        if last_exc:
            raise last_exc
        return ""


class ExplainerModule:
    def __init__(self, backend: LLMBackend = None):
        if backend is None:
            # Auto-detect local Ollama server first, then Groq, then MockBackend fallback
            try:
                req = urllib.request.Request("http://localhost:11434/api/tags", headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=2) as resp:
                    tags = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("name", "") for m in tags.get("models", [])]
                    local_model = models[0] if models else "qwen3.5:4b"
                    self.backend = OllamaBackend(model=local_model)
            except Exception:
                if "GROQ_API_KEY" in os.environ:
                    self.backend = GroqBackend()
                else:
                    self.backend = MockBackend()
        else:
            self.backend = backend

        self.validator = GroundingValidator()

    def build_prompt(self, evidence: dict) -> str:
        sf = evidence["state_facts"]
        pe = evidence["policy_evidence"]
        ctx = evidence["context"]

        deck_sz = int(sf["deck_size"])
        danger_ratio = f"1 in {max(1, deck_sz)}"

        top_alts_str = ", ".join([f"{item['action']} ({item['probability']:.1%})" for item in pe["top_legal_actions"]])
        hand_str = ", ".join([f"{k}: {v}" for k, v in sf["own_hand_counts"].items()]) or "empty"

        alts_example = json.dumps([
            {"action": item["action"], "probability": round(float(item["probability"]), 3)}
            for item in pe["top_legal_actions"]
            if item["action"] != pe["chosen_action_name"]
        ][:3])

        act_name = pe["chosen_action_name"]
        act_prob_str = f"{pe['chosen_action_probability']:.1%}"
        v_s_str = f"{pe['value_estimate_V_s']:.2f}"

        prompt = (
            "ROLE:\n"
            "You are the strategic explanation engine for 'StratDeck', an adversarial reinforcement learning (RL) survival card game.\n"
            "The agent is a trained MaskablePPO policy operating under strict action masking.\n\n"
            "GAME RULES & TACTICAL MECHANICS:\n"
            "- Core Objective: Survive and eliminate opponents. The draw deck contains a lethal instant-elimination 'HITMAN' bomb card.\n"
            f"- Drawing cards carries lethal risk: current deck size is {deck_sz} cards (odds of drawing Hitman are roughly {danger_ratio}).\n"
            "- SHIELD: Automatically defuses a drawn HITMAN card (saving player's life), or can be played to skip a draw.\n"
            "- ATTACK: Concludes turn without drawing, forcing the opponent to take 2 consecutive turns (forcing them into dangerous draws).\n"
            "- SKIP / REVERSE: Safely concludes turn without drawing from the deck.\n"
            "- TRAP: Sets a penalty forcing the opponent to draw an extra card on their next draw.\n"
            "- SPY / STEAL / SWAP: Free utility cards (0-cost, turn continues) that provide recon, theft, or card exchange.\n"
            "- DRAW: Concludes turn by drawing 1 card from the deck. Expands hand resources, but risks instant elimination if HITMAN is drawn.\n\n"
            "TASK:\n"
            f"Generate a comprehensive, insightful, and detailed tactical explanation (3 to 5 full sentences) explaining why the RL agent chose {act_name} in this exact situation.\n\n"
            "EVIDENCE:\n"
            f"- Selected Action: {act_name} (Policy Probability: {act_prob_str}, Rank: #{pe['chosen_action_rank']})\n"
            f"- Available Legal Actions & Policy Probabilities: {top_alts_str}\n"
            f"- Agent State-Value Estimate V(s): {v_s_str} (expected match return)\n"
            f"- Agent Current Hand: {hand_str}\n"
            f"- Deck Size: {deck_sz} cards (Hitman odds: {danger_ratio})\n"
            f"- Pending Forced Draws on Agent: {sf['pending_draw_count']}\n"
            f"- Active Players Remaining: {sf['alive_players_count']}\n"
            f"- Opponent Behavioral Profile: Cluster {ctx.get('profiler_cluster', 'N/A')}\n\n"
            "STRICT RULES:\n"
            f"1. Detail the strategic rationale of choosing {act_name} in relation to the current deck size ({deck_sz}), Hitman risk ({danger_ratio}), and hand resources.\n"
            "2. Compare it to the legal alternatives (e.g., why this action was prioritized over other options).\n"
            f"3. Mention the exact policy probability ({act_prob_str}) and state-value V(s) ({v_s_str}).\n"
            f"4. IMPORTANT: Do NOT use the '%' symbol for anything other than the exact action probabilities provided above (e.g., write 'odds are {danger_ratio}' instead of a percentage for the deck danger).\n"
            "5. FORBIDDEN WORDS: Do NOT use words like 'determined', 'knew', 'optimal', 'sole viable strategy', 'caused the action', 'Q-value', or recommend alternative moves.\n\n"
            "OUTPUT FORMAT (Return valid JSON ONLY, no extra text):\n"
            "{\n"
            '  "selected_action": "' + act_name + '",\n'
            '  "explanation": "<Detailed 3-5 sentence tactical explanation grounded in StratDeck game rules and policy evidence>",\n'
            '  "evidence": ["Selected action ' + act_name + ' with policy probability ' + act_prob_str + '", "Current deck size ' + str(deck_sz) + ' with ' + danger_ratio + ' Hitman odds", "State-value estimate V(s) is ' + v_s_str + '"],\n'
            '  "legal_alternatives": ' + alts_example + ',\n'
            '  "grounding_status": "SUPPORTED"\n'
            "}"
        )
        return prompt

    def explain_move(self, evidence: dict) -> dict:
        prompt = self.build_prompt(evidence)
        raw_response = self.backend.generate(prompt)

        # Parse JSON response
        try:
            explanation_data = json.loads(raw_response)
        except Exception:
            explanation_data = {
                "selected_action": evidence["policy_evidence"]["chosen_action_name"],
                "explanation": raw_response,
                "evidence": ["Raw text response received"],
                "legal_alternatives": evidence["policy_evidence"]["top_legal_actions"],
                "grounding_status": "UNPARSED"
            }

        # Run automatic Grounding Validator
        validation_result = self.validator.validate(evidence, explanation_data)
        explanation_data["validation"] = validation_result

        return explanation_data
