"""
RenderCallback — visualisation PÉRIODIQUE pendant l'entraînement
================================================================

Toutes les `render_freq` étapes, met l'entraînement en pause, joue la
politique courante (déterministe) dans une fenêtre pygame, puis reprend.
-> on voit la marche se développer EN DIRECT, sans attendre la fin.

Les statistiques de normalisation sont synchronisées depuis l'environnement
d'entraînement à chaque visualisation, pour un rendu fidèle.

Utilisé par train.py via l'option --watch-every.
"""

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv, VecNormalize, sync_envs_normalization,
)

from quadruped_env import QuadrupedEnv
import creature_env as ce


class RenderCallback(BaseCallback):
    def __init__(self, render_freq, with_obstacles=False, n_episodes=1,
                 max_seconds=10.0, difficulty=1.0, rough_terrain=False, verbose=1):
        super().__init__(verbose)
        self.render_freq = int(render_freq)
        self.with_obstacles = with_obstacles
        self.difficulty = difficulty
        self.rough_terrain = rough_terrain
        self.n_episodes = n_episodes
        self.max_steps = int(max_seconds * ce.FPS)
        self._next = 0
        self.render_env = None

    def _on_training_start(self):
        base = DummyVecEnv([lambda: QuadrupedEnv(
            with_obstacles=self.with_obstacles, difficulty=self.difficulty,
            rough_terrain=self.rough_terrain, render_mode="human")])
        # training=False : on ne met pas à jour les stats, on les synchronise
        self.render_env = VecNormalize(base, training=False,
                                       norm_obs=True, norm_reward=False)
        # première visualisation après render_freq pas (en repartant du compteur courant)
        self._next = self.model.num_timesteps + self.render_freq

    def _on_step(self):
        self._pump_events()           # garde la fenêtre réactive entre 2 visus
        if self.num_timesteps >= self._next:
            self._next += self.render_freq
            self._play()
        return True

    def _pump_events(self):
        try:
            import pygame
            if pygame.display.get_init():
                pygame.event.pump()
        except Exception:
            pass

    def _play(self):
        # aligne la normalisation du rendu sur celle de l'entraînement
        sync_envs_normalization(self.training_env, self.render_env)
        self.render_env.env_method("set_banner",
                                   f"entrainement : {self.num_timesteps:,} pas")
        for _ in range(self.n_episodes):
            obs = self.render_env.reset()
            done = [False]
            steps = 0
            while not done[0] and steps < self.max_steps:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _ = self.render_env.step(action)
                steps += 1
        if self.verbose:
            print(f"[viz] politique affichee a {self.num_timesteps:,} pas")

    def _on_training_end(self):
        if self.render_env is not None:
            self.render_env.close()
            self.render_env = None
