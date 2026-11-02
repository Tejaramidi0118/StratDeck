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

            for line in lines:
                if "- Selected Action:" in line:
                    chosen_action = line.split("- Selected Action:")[1].strip().split()[0]
                elif "- Selected Action Probability:" in line:
                    p_str = line.split("- Selected Action Probability:")[1].strip().replace("%", "")
                    chosen_prob = float(p_str) / 100.0 if float(p_str) > 1.0 else float(p_str)
                elif "- Value Estimate V(s):" in line:
                    v_s = float(line.split("- Value Estimate V(s):")[1].strip())
                elif "- Top Legal Actions:" in line:
                    # Parse "ATTACK (61.0%), SHIELD (18.0%), SPY (12.0%)"
                    raw_alts = line.split("- Top Legal Actions:")[1].strip()
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
                    f"The RL agent selected {chosen_action} with a policy probability of {chosen_prob:.1%}, "
                    f"which was the top-ranked choice among legal actions. Legal alternatives considered: {alt_str}. "
                    f"Value estimate V(s) is {v_s:.2f}."
                ),
                "evidence": [
                    f"Selected action {chosen_action} has policy probability {chosen_prob:.1%}",
                    f"Legal alternatives considered: {alt_str}",
                    f"Value estimate V(s) is {v_s:.2f}"
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
                "num_predict": 300,
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

        top_alts_str = ", ".join([f"{item['action']} ({item['probability']:.1%})" for item in pe["top_legal_actions"]])
        hand_str = ", ".join([f"{k}: {v}" for k, v in sf["own_hand_counts"].items()]) or "empty"

        prompt = (
            "ROLE:\n"
            "You are an explanation generator for a reinforcement-learning card-game agent.\n\n"
            "TASK:\n"
            "Explain the ALREADY-SELECTED action using ONLY the supplied factual evidence.\n\n"
            "EVIDENCE HIERARCHY:\n"
            "1. Level 1 (Primary Policy Evidence): Selected action, selected action probability pi(a|s), rank, and probabilities of legal alternatives.\n"
            "2. Level 2 (Observable State): Own hand, legal actions, deck size, pending forced draws, alive players.\n"
            "3. Level 3 (Value Estimate): V(s) is the model's state-value estimate. Describe it only as state value; DO NOT claim V(s) caused the action selection or infer Q-values.\n\n"
            "STRICT RULES:\n"
            "- Do NOT claim V(s) directly caused the action selection.\n"
            "- Do NOT say 'the agent knew', 'the agent determined', 'the opponent probably', 'sole viable strategy', 'only viable strategy', 'action was optimal', or 'had to choose'.\n"
            "- Do NOT invent opponent hidden cards, deck order, or unsupplied facts.\n"
            "- Explain primarily using Level 1 and Level 2 evidence in an objective, post-hoc natural language style.\n\n"
            "EVIDENCE:\n"
            f"- Selected Action: {pe['chosen_action_name']} (Index {pe['chosen_action_idx']})\n"
            f"- Selected Action Probability: {pe['chosen_action_probability']:.1%}\n"
            f"- Selected Action Rank: {pe['chosen_action_rank']} of {len(sf['legal_actions'])}\n"
            f"- Top Legal Actions: {top_alts_str}\n"
            f"- Value Estimate V(s): {pe['value_estimate_V_s']:.2f}\n"
            f"- Current Hand: {hand_str}\n"
            f"- Deck Size: {sf['deck_size']}\n"
            f"- Pending Forced Draws: {sf['pending_draw_count']}\n"
            f"- Alive Players: {sf['alive_players_count']}\n"
            f"- DDA Tier Context: {ctx.get('dda_difficulty_level', 'N/A')}\n"
            f"- Behavioral Profile Context: Cluster {ctx.get('profiler_cluster', 'N/A')}\n\n"
            "OUTPUT FORMAT (JSON ONLY):\n"
            "{\n"
            '  "selected_action": "' + pe['chosen_action_name'] + '",\n'
            '  "explanation": "<2-3 sentence objective explanation grounded in Level 1 & 2 evidence>",\n'
            '  "evidence": ["<bullet point 1>", "<bullet point 2>"],\n'
            '  "legal_alternatives": [{"action": "<name>", "probability": <float>}],\n'
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
