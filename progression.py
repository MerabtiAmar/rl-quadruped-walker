"""
Lecteur de progression de l'apprentissage
==========================================

Rejoue, dans l'ordre chronologique, tous les checkpoints sauvegardés pendant
l'entraînement -> on voit la créature évoluer du chaos initial jusqu'à la
marche, dans une seule et même fenêtre.

Chaque checkpoint utilise SES propres statistiques de normalisation (le .pkl
correspondant), pour un rendu fidèle à ce que la politique "voyait" à ce
moment de l'entraînement.

Lancement :
  python progression.py                         # tous les checkpoints de ppo_quadruped
  python progression.py --name ppo_quadruped --episodes 1
  python progression.py --obstacles             # si l'entraînement avait les obstacles
  python progression.py --max-seconds 8         # limite le temps par checkpoint

Pendant la lecture : ferme la fenêtre pour arrêter.
"""

import argparse
import glob
import os
import re

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from quadruped_env import QuadrupedEnv
import creature_env as ce


def find_checkpoints(name):
    """Retourne [(steps, model_path, vecnorm_path), ...] trié par steps."""
    pattern = os.path.join("models", f"{name}_*_steps.zip")
    items = []
    for model_path in glob.glob(pattern):
        m = re.search(r"_(\d+)_steps\.zip$", model_path)
        if not m:
            continue
        steps = int(m.group(1))
        vec = os.path.join("models", f"{name}_vecnormalize_{steps}_steps.pkl")
        if not os.path.exists(vec):
            vec = os.path.join("models", f"{name}_vecnormalize.pkl")  # repli
        items.append((steps, model_path, vec))
    # ajoute le modèle final s'il existe et n'est pas déjà couvert
    items.sort(key=lambda x: x[0])
    return items


def play_checkpoint(base_env, model_path, vecnorm_path, steps, episodes, max_steps):
    """Joue une politique sur l'environnement de rendu partagé `base_env`."""
    env = base_env
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, base_env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(model_path, device="cpu")
    base_env.env_method("set_banner", f"{steps:,} pas")

    ep, played, total = 0, 0, 0.0
    obs = env.reset()
    while ep < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total += float(reward[0])
        played += 1
        if done[0] or played >= max_steps:
            cause = ("cible" if info[0].get("reached")
                     else "chute" if info[0].get("fell") else "timeout/limite")
            print(f"  {steps:>8,} pas : recompense={total:8.1f}  fin={cause}")
            ep += 1
            played, total = 0, 0.0
            obs = env.reset()


def main():
    parser = argparse.ArgumentParser(description="Rejouer la progression de l'entraînement")
    parser.add_argument("--name", default="ppo_quadruped")
    parser.add_argument("--obstacles", action="store_true")
    parser.add_argument("--terrain", action="store_true",
                        help="afficher avec terrain ondulé (doit matcher l'entraînement)")
    parser.add_argument("--episodes", type=int, default=1,
                        help="épisodes joués par checkpoint")
    parser.add_argument("--max-seconds", type=float, default=10.0,
                        help="durée max d'un épisode par checkpoint")
    args = parser.parse_args()

    checkpoints = find_checkpoints(args.name)
    if not checkpoints:
        raise SystemExit(f"Aucun checkpoint trouvé pour '{args.name}' dans models/.\n"
                         f"As-tu lancé l'entraînement ?")

    print(f"{len(checkpoints)} checkpoints trouvés. Lecture de la progression...\n")

    # UNE seule fenêtre partagée pour toute la progression
    base_env = DummyVecEnv([lambda: QuadrupedEnv(with_obstacles=args.obstacles,
                                                 rough_terrain=args.terrain,
                                                 render_mode="human")])
    max_steps = int(args.max_seconds * ce.FPS)

    try:
        for steps, model_path, vecnorm_path in checkpoints:
            play_checkpoint(base_env, model_path, vecnorm_path, steps,
                            args.episodes, max_steps)
    finally:
        base_env.close()

    print("\nProgression terminée.")


if __name__ == "__main__":
    main()
