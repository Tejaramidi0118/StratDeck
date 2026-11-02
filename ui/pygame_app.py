"""
Phase 8: Minimal Pygame UI for human-vs-AI play.
Run: python -m ui.pygame_app
Click a card in your hand to play it, or click "DRAW" to draw.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
from stable_baselines3 import PPO
from env.gym_wrapper import AgentZeroEnv
from env.cards import Card
from modules.explainer import ExplainerModule, OllamaBackend

WIDTH, HEIGHT = 900, 500
FONT_SIZE = 20
CARD_W, CARD_H = 110, 60
MODEL_PATH = "checkpoints/ppo/ppo_final"


def draw_text(screen, font, text, x, y, color=(255, 255, 255)):
    surf = font.render(text, True, color)
    screen.blit(surf, (x, y))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Agent Zero")
    font = pygame.font.SysFont("arial", FONT_SIZE)
    clock = pygame.time.Clock()

    env = AgentZeroEnv(num_players=2, opponent_policy=None)
    # human is player 1 (index 1); agent (index 0) is controlled by the trained model
    env.agent_player_id = 0

    model = None
    if os.path.exists(MODEL_PATH + ".zip"):
        model = PPO.load(MODEL_PATH)

    explainer = ExplainerModule(backend=OllamaBackend(model="llama3.1"))

    obs, _ = env.reset()
    last_explanation = ""
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                human_id = 1
                legal = env.engine.legal_actions(human_id)
                for i, card_action in enumerate(legal):
                    cx = 20 + i * (CARD_W + 10)
                    cy = HEIGHT - 100
                    if cx <= mx <= cx + CARD_W and cy <= my <= cy + CARD_H:
                        if env.engine.current_player == human_id:
                            env.engine.step(card_action)

        # let the trained agent act if it's its turn
        if not env.engine.game_over and env.engine.current_player == 0 and model is not None:
            action, _ = model.predict(obs, deterministic=True)
            card_action = env._decode_action(int(action))
            legal = env.engine.legal_actions(0)
            if card_action not in legal:
                card_action = env.engine.DRAW_ACTION
            env.engine.step(card_action)
            obs = env._encode_obs()

            if explainer is not None and model is not None:
                try:
                    import torch
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(model.policy.device)
                    mask_tensor = torch.as_tensor(env.action_masks(), dtype=torch.bool).unsqueeze(0).to(model.policy.device)
                    with torch.no_grad():
                        dist = model.policy.get_distribution(obs_tensor, action_masks=mask_tensor)
                        real_probs = dist.distribution.probs.cpu().numpy()[0]
                    prob_dict = {a: float(real_probs[a]) for a in range(len(real_probs)) if real_probs[a] > 0.01}
                    deck_sz = len(env.engine.deck)
                    hitman_p = round(1.0 / max(1, deck_sz), 2)
                    last_explanation = explainer.explain_move(
                        chosen_action=card_action,
                        action_probs=prob_dict,
                        hand=env.engine.players[0].hand,
                        hitman_probability=hitman_p,
                    )
                except Exception as e:
                    last_explanation = f"(explanation fallback: {e})"

        # --- render ---
        screen.fill((20, 30, 40))
        draw_text(screen, font, f"Deck size: {len(env.engine.deck)}", 20, 20)
        draw_text(screen, font, f"Current turn: Player {env.engine.current_player}", 20, 50)

        human = env.engine.players[1]
        draw_text(screen, font, "Your hand (click to play):", 20, HEIGHT - 130)
        for i, card in enumerate(human.hand):
            cx = 20 + i * (CARD_W + 10)
            cy = HEIGHT - 100
            pygame.draw.rect(screen, (80, 100, 140), (cx, cy, CARD_W, CARD_H))
            draw_text(screen, font, Card(card).name, cx + 5, cy + 20)

        draw_text(screen, font, "AI's last move explanation:", 20, 100)
        draw_text(screen, font, last_explanation[:100], 20, 130, color=(200, 220, 100))

        if env.engine.game_over:
            draw_text(screen, font, f"GAME OVER - Winner: Player {env.engine.winner}", 300, 250, color=(255, 80, 80))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
