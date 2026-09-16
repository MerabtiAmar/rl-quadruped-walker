"""
Visualisation d'une politique entraînée
=======================================

Charge un modèle PPO sauvegardé (+ les statistiques de normalisation) et le
fait jouer dans la fenêtre pygame, en mode déterministe (la meilleure action,
sans bruit d'exploration).

Lancement :
  python watch.py                              # modèle ppo_quadruped (marche)
  python watch.py --name ppo_quadruped --obstacles
  python watch.py --model models/ppo_quadruped_120000_steps.zip   # un checkpoint précis
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from quadruped_env import QuadrupedEnv


def main():
    parser = argparse.ArgumentParser(description="Visualiser une politique PPO")
    parser.add_argument("--name", default="ppo_quadruped",
                        help="nom de l'expérience (pour retrouver les fichiers)")
    parser.add_argument("--model", default=None,
                        help="chemin explicite vers un .zip (sinon <name>_final)")
    parser.add_argument("--vecnorm", default=None,
                        help="chemin explicite vers le .pkl de normalisation")
    parser.add_argument("--obstacles", action="store_true",
                        help="afficher avec obstacles (doit matcher l'entraînement)")
    parser.add_argument("--terrain", action="store_true",
                        help="afficher avec terrain ondulé (doit matcher l'entraînement)")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    model_path = args.model or os.path.join("models", f"{args.name}_final.zip")
    vecnorm_path = args.vecnorm or os.path.join("models", f"{args.name}_vecnormalize.pkl")

    if not os.path.exists(model_path):
        raise SystemExit(f"Modèle introuvable : {model_path}\n"
                         f"As-tu lancé l'entraînement (python train.py) ?")

    # Environnement de rendu, enveloppé comme à l'entraînement
    env = DummyVecEnv([lambda: QuadrupedEnv(with_obstacles=args.obstacles,
                                            rough_terrain=args.terrain,
                                            render_mode="human")])
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False        # ne plus mettre à jour les stats
        env.norm_reward = False     # on veut la vraie récompense
    else:
        print(f"(info) stats de normalisation non trouvées ({vecnorm_path}) "
              f"-> rendu sans normalisation.")

    model = PPO.load(model_path, device="cpu")

    print(f"Modèle : {model_path}")
    print("Lecture en cours (ferme la fenêtre ou Ctrl+C pour arrêter)...")

    episode, total = 0, 0.0
    obs = env.reset()
    while episode < args.episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total += float(reward[0])
        if done[0]:
            episode += 1
            cause = ("cible" if info[0].get("reached")
                     else "chute" if info[0].get("fell") else "timeout")
            print(f"épisode {episode}: récompense={total:8.1f}  fin={cause}")
            total = 0.0
            # VecEnv se réinitialise automatiquement après un 'done'

    env.close()


if __name__ == "__main__":
    main()
