"""
Entraînement PPO du quadrupède
==============================

Branche l'algorithme PPO (stable-baselines3) sur l'environnement
`QuadrupedEnv`. Setup "sérieux" :

  - Environnements PARALLÈLES (SubprocVecEnv) : plusieurs créatures simulées
    en même temps -> collecte de données beaucoup plus rapide.
  - VecNormalize : normalise observations et récompenses à la volée
    (running mean/std). Quasi indispensable pour stabiliser PPO en contrôle
    continu.
  - Checkpoints réguliers (modèle + stats de normalisation).
  - Logs TensorBoard pour suivre la courbe d'apprentissage en direct.

Hyperparamètres : valeurs éprouvées pour la locomotion (proches du "zoo" SB3).

Lancement (voir le bas du fichier et les explications fournies) :
  python train.py                          # 1M pas, sans obstacles (marche)
  python train.py --timesteps 3000000      # entraînement plus long
  python train.py --obstacles              # tâche complète (marche + évitement)
  python train.py --n-envs 8               # + d'environnements parallèles
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from quadruped_env import QuadrupedEnv
from render_callback import RenderCallback


# Hyperparamètres PPO (réglages classiques de locomotion continue)
HYPERPARAMS = dict(
    learning_rate=3e-4,      # pas d'apprentissage Adam
    n_steps=2048,            # pas collectés PAR environnement avant chaque update
    batch_size=256,          # taille des mini-batchs de la descente de gradient
    n_epochs=10,             # nb de passes sur les données à chaque cycle
    gamma=0.99,              # actualisation (horizon ~100 pas)
    gae_lambda=0.95,         # compromis biais/variance de GAE
    clip_range=0.2,          # le fameux "epsilon" du clipping PPO
    ent_coef=0.0,            # bonus d'entropie (exploration)
    vf_coef=0.5,             # poids de la loss du critic
    max_grad_norm=0.5,       # clipping du gradient (stabilité)
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),  # 2 réseaux 256x256
)

MODELS_DIR = "models"
LOGS_DIR = "logs"


def main():
    parser = argparse.ArgumentParser(description="Entraînement PPO du quadrupède")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="nombre total de pas d'environnement")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="nombre d'environnements parallèles")
    parser.add_argument("--obstacles", action="store_true",
                        help="activer les obstacles (tâche complète)")
    parser.add_argument("--name", default="ppo_quadruped",
                        help="nom de l'expérience (fichiers + TensorBoard)")
    parser.add_argument("--watch-every", type=int, default=0,
                        help="visualiser la politique toutes les N étapes (0 = jamais)")
    parser.add_argument("--difficulty", type=float, default=1.0,
                        help="sévérité MAX des obstacles et du relief (0..1)")
    parser.add_argument("--terrain", action="store_true",
                        help="activer le terrain ondulé (relief lisse et continu)")
    parser.add_argument("--continue", dest="resume", action="store_true",
                        help="reprendre l'entraînement depuis <name>_final.zip")
    parser.add_argument("--init-from", default=None,
                        help="démarrer une NOUVELLE expérience à partir des poids "
                             "d'un autre modèle (nom d'expérience, ex. ppo_quadruped). "
                             "Idéal pour le curriculum : marche -> obstacles.")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, f"{args.name}_final.zip")
    vecnorm_path = os.path.join(MODELS_DIR, f"{args.name}_vecnormalize.pkl")
    resuming = args.resume and os.path.exists(model_path)

    # Source de transfert (curriculum) : poids + normalisation d'un autre modèle
    init_model = init_vecnorm = None
    if args.init_from and not resuming:
        init_model = os.path.join(MODELS_DIR, f"{args.init_from}_final.zip")
        init_vecnorm = os.path.join(MODELS_DIR, f"{args.init_from}_vecnormalize.pkl")
        if not os.path.exists(init_model):
            raise SystemExit(f"Modèle d'initialisation introuvable : {init_model}")

    # --- Environnements parallèles
    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    raw_env = make_vec_env(
        QuadrupedEnv, n_envs=args.n_envs,
        env_kwargs=dict(with_obstacles=args.obstacles, difficulty=args.difficulty,
                        rough_terrain=args.terrain),
        vec_env_cls=vec_cls,
    )

    # --- Normalisation : reprise/transfert des stats, sinon neuves
    if resuming and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, raw_env)
        env.training = True
        env.norm_reward = True
    elif init_vecnorm and os.path.exists(init_vecnorm):
        env = VecNormalize.load(init_vecnorm, raw_env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # --- Le modèle PPO (CPU : plus rapide que GPU pour un si petit réseau)
    if resuming:
        model = PPO.load(model_path, env=env, device="cpu",
                         tensorboard_log=LOGS_DIR)
        print(f"(reprise depuis {model_path} a {model.num_timesteps:,} pas)")
    elif init_model:
        model = PPO.load(init_model, env=env, device="cpu",
                         tensorboard_log=LOGS_DIR)
        print(f"(transfert depuis {init_model} -> nouvelle experience '{args.name}')")
    else:
        model = PPO("MlpPolicy", env, device="cpu", verbose=1,
                    tensorboard_log=LOGS_DIR, **HYPERPARAMS)

    # --- Callbacks : sauvegardes + visualisation périodique optionnelle
    callbacks = [CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path=MODELS_DIR,
        name_prefix=args.name,
        save_vecnormalize=True,
    )]
    if args.watch_every > 0:
        callbacks.append(RenderCallback(
            render_freq=args.watch_every,
            with_obstacles=args.obstacles,
            difficulty=args.difficulty,
            rough_terrain=args.terrain,
        ))

    print("=" * 64)
    print(f"  Entraînement PPO : {args.name}{'  (REPRISE)' if resuming else ''}")
    print(f"  Pas (ce run)     : {args.timesteps:,}")
    print(f"  Environnements   : {args.n_envs} en parallèle ({vec_cls.__name__})")
    print(f"  Obstacles        : {'OUI (tâche complète)' if args.obstacles else 'NON (marche)'}")
    print(f"  Visualisation    : {('toutes les ' + format(args.watch_every, ',') + ' etapes') if args.watch_every else 'non'}")
    print(f"  TensorBoard      : tensorboard --logdir {LOGS_DIR}")
    print("=" * 64)

    model.learn(total_timesteps=args.timesteps,
                callback=CallbackList(callbacks),
                tb_log_name=args.name,
                reset_num_timesteps=not resuming)

    # --- Sauvegarde finale
    model.save(os.path.join(MODELS_DIR, f"{args.name}_final"))
    env.save(os.path.join(MODELS_DIR, f"{args.name}_vecnormalize.pkl"))
    print("\nEntraînement terminé.")
    print(f"Modèle      -> {MODELS_DIR}/{args.name}_final.zip")
    print(f"Normalisat. -> {MODELS_DIR}/{args.name}_vecnormalize.pkl")
    print(f"Visualiser  -> python watch.py --name {args.name}"
          + (" --obstacles" if args.obstacles else ""))


if __name__ == "__main__":
    main()
