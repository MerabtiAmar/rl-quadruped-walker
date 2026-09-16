"""
Environnement d'apprentissage par renforcement — Quadruped (Gymnasium)
======================================================================

Enveloppe la physique de `creature_env.py` dans l'interface standard
Gymnasium (`reset` / `step`), pour pouvoir y brancher n'importe quel algo
de Deep RL (PPO, SAC...). C'est le "terrain d'entraînement".

Le MDP (processus de décision) :

  ACTION  (ce que l'agent décide)
      8 réels dans [-1, 1] -> vitesses des 8 moteurs (hanche/genou x4 pattes).

  OBSERVATION (ce que l'agent perçoit) — vecteur de 28 réels :
      - torse : sin/cos de l'angle, vitesse angulaire, vitesse (x,y), hauteur
      - 4 pattes x (angle hanche, vit. hanche, angle genou, vit. genou) = 16
      - distance horizontale à la cible
      - 5 capteurs de distance (raycasts) vers l'avant/le bas (sol + obstacles)

  RÉCOMPENSE (reward shaping) :
      + progression vers la cible      (terme principal : marcher dans le bon sens)
      + bonus de survie                (rester "en vie" / debout)
      - inclinaison du torse           (encourage l'équilibre)
      - énergie dépensée               (mouvements plus propres)
      - collision avec un obstacle     (si obstacles activés)
      + gros bonus si cible atteinte
      - pénalité si chute

  FIN D'ÉPISODE :
      terminated : cible atteinte OU chute (torse trop incliné / trop bas)
      truncated  : nombre max de pas atteint (timeout)

Curriculum conseillé :
  1) with_obstacles=False  -> apprendre à marcher vers la cible
  2) with_obstacles=True   -> ajouter l'évitement une fois la marche acquise

Utilisation :
  python quadruped_env.py            # rollout aléatoire (headless) + stats
  python quadruped_env.py --check    # validation Gymnasium de l'environnement
  python quadruped_env.py --render   # visualiser une politique ALÉATOIRE
"""

import sys
import math
import random
import numpy as np
import pymunk
import gymnasium as gym
from gymnasium import spaces

import creature_env as ce

# Types de collision (pour détecter créature <-> obstacle)
COL_CREATURE = 1
COL_OBSTACLE = 2


class QuadrupedEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": ce.FPS}

    def __init__(self, with_obstacles=False, max_steps=1200, render_mode=None,
                 difficulty=1.0, rough_terrain=False):
        super().__init__()
        self.with_obstacles = with_obstacles
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.max_difficulty = difficulty     # sévérité MAX des obstacles randomisés
        self.rough_terrain = rough_terrain   # terrain ondulé (True) ou plat (False)

        # --- Paramètres de contrôle / physique
        self.MAX_RATE = 6.0                  # rad/s pour |action| = 1
        self.n_rays = 5
        self.spawn_clearance = 78            # hauteur d'apparition au-dessus du sol
        self.target_radius = 40.0
        self.fall_angle = math.pi / 2        # chute déclarée au-delà de 90°
        self._fall_grace_max = int(2.0 * ce.FPS)   # sursis animation : 2 s après 90°
        self._fall_grace = self._fall_grace_max
        self._obs_rng = random.Random()      # RNG dédié à la randomisation
        self.terrain = None                  # relief courant (None = plat)

        # --- Espace d'ACTION : 8 moteurs normalisés
        self.action_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)

        # --- Espace d'OBSERVATION : 28 réels (bornes finies larges)
        obs_dim = 2 + 1 + 2 + 1 + 16 + 1 + self.n_rays
        high = np.full(obs_dim, 1e3, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # --- État interne
        self.space = None
        self.creature = None
        self.obstacles = None
        self.steps = 0
        self.prev_dist = 0.0
        self._collided = False

        # --- Rendu (initialisé paresseusement)
        self._screen = None
        self._clock = None
        self._font = None
        self._bigfont = None
        self._field = None
        self.banner = ""        # texte optionnel affiché en haut (ex. checkpoint)

    def set_banner(self, text):
        """Définit le bandeau affiché en haut de la fenêtre (utilisé par
        progression.py pour indiquer le checkpoint en cours)."""
        self.banner = text

    # ------------------------------------------------------------------ API gym
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Domain randomization : nouvelle disposition à chaque épisode + difficulté
        # échantillonnée (mélange de cas faciles et difficiles -> robustesse).
        if self.with_obstacles:
            ep_difficulty = self.max_difficulty * float(self.np_random.uniform(0.25, 1.0))
        else:
            ep_difficulty = 0.0
        self._obs_rng.seed(int(self.np_random.integers(0, 2**31 - 1)))

        self.space, self.creature, self.obstacles, self.terrain = ce.build_world(
            with_obstacles=self.with_obstacles, spawn_clearance=self.spawn_clearance,
            rng=self._obs_rng, difficulty=ep_difficulty,
            rough_terrain=self.rough_terrain)

        # Collision pénalisée UNIQUEMENT pour le torse : les pattes DOIVENT
        # pouvoir prendre appui sur un obstacle pour grimper (elles entrent
        # physiquement en collision dans tous les cas, sans déclencher la pénalité).
        list(self.creature.torso.shapes)[0].collision_type = COL_CREATURE
        for _, os in self.obstacles:
            os.collision_type = COL_OBSTACLE
        if self.with_obstacles:
            self.space.on_collision(COL_CREATURE, COL_OBSTACLE,
                                    pre_solve=self._mark_collision)

        self.steps = 0
        self._collided = False
        self._fall_grace = self._fall_grace_max
        self.prev_dist = self._dist_to_target()
        return self._get_obs(), {}

    def _mark_collision(self, arbiter, space, data):
        """Callback pymunk : une pièce de la créature touche un obstacle."""
        self._collided = True

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.creature.set_motor_rates([float(a) * self.MAX_RATE for a in action])

        # avance la physique (mêmes sous-pas que le visualiseur, pour la stabilité)
        dt = 1.0 / ce.FPS
        self._collided = False
        for _ in range(6):
            self.space.step(dt / 6)
        self.steps += 1

        obs = self._get_obs()
        reward, terminated, info = self._reward_done(action)
        truncated = self.steps >= self.max_steps
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    # -------------------------------------------------------------- observation
    def _dist_to_target(self):
        return ce.TARGET_X - self.creature.torso.position.x

    def _ground_y(self, x):
        """Hauteur du sol (relief) sous l'abscisse x."""
        return ce.terrain_height(self.terrain, x)

    def _get_obs(self):
        torso = self.creature.torso
        ang = torso.angle
        # hauteur du torse AU-DESSUS du sol local (valable même en terrain ondulé)
        height = self._ground_y(torso.position.x) - torso.position.y
        obs = [
            math.sin(ang), math.cos(ang),
            torso.angular_velocity,
            torso.velocity.x / 200.0, torso.velocity.y / 200.0,
            height / 100.0,
        ]
        for leg in self.creature.legs:
            thigh, shin = leg["thigh"], leg["shin"]
            obs += [
                thigh.angle - torso.angle,
                thigh.angular_velocity - torso.angular_velocity,
                shin.angle - thigh.angle,
                shin.angular_velocity - thigh.angular_velocity,
            ]
        obs.append(self._dist_to_target() / 500.0)
        obs += self._raycasts()
        return np.array(obs, dtype=np.float32)

    def _raycasts(self):
        """Distances libres (0..1) le long de rayons vers l'avant/le bas."""
        origin = self.creature.torso.position
        length = 220.0
        filt = pymunk.ShapeFilter(group=ce.CREATURE_GROUP)  # ignore la créature
        out = []
        for a in np.linspace(-0.2, 1.2, self.n_rays):       # 0=avant, +=vers le bas
            end = (origin.x + math.cos(a) * length,
                   origin.y + math.sin(a) * length)
            hit = self.space.segment_query_first(origin, end, 1, filt)
            out.append(float(hit.alpha) if hit is not None else 1.0)
        return out

    # ------------------------------------------------------------ reward & done
    def _reward_done(self, action):
        torso = self.creature.torso
        dist = self._dist_to_target()
        progress = self.prev_dist - dist     # > 0 si la créature avance vers la cible
        self.prev_dist = dist

        ang = (torso.angle + math.pi) % (2 * math.pi) - math.pi   # angle dans [-pi, pi]
        height = self._ground_y(torso.position.x) - torso.position.y  # au-dessus du sol local

        reward = (
            1.5 * progress                       # progression (terme dominant)
            + 0.3                                # bonus de survie
            - 0.5 * abs(ang)                     # rester droit
            - 0.003 * float(np.sum(action ** 2)) # économie d'énergie
        )

        terminated = False
        info = {}

        if self.with_obstacles and self._collided:
            reward -= 1.0
            info["collision"] = True

        if dist < self.target_radius:
            reward += 200.0
            terminated = True
            info["reached"] = True

        # Chute : torse au-delà de 90° OU torse écrasé au sol
        fallen = (abs(ang) > self.fall_angle) or (height < 18)
        if not terminated and fallen:
            if self.render_mode == "human":
                # Animation : on laisse 2 s après le franchissement de 90° (pour
                # voir une éventuelle récupération) avant d'arrêter l'épisode.
                self._fall_grace -= 1
                if self._fall_grace <= 0:
                    reward -= 10.0
                    terminated = True
                    info["fell"] = True
            else:
                # Entraînement : arrêt immédiat à 90° (plus de liberté qu'avant).
                reward -= 10.0
                terminated = True
                info["fell"] = True
        elif not fallen and self.render_mode == "human":
            self._fall_grace = self._fall_grace_max   # rétablissement -> sursis réarmé

        return reward, terminated, info

    # ------------------------------------------------------------------- render
    def render(self):
        if self.render_mode != "human":
            return
        import pygame
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((ce.WIDTH, ce.HEIGHT))
            pygame.display.set_caption("Quadruped — environnement RL")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("consolas", 16)
            self._bigfont = pygame.font.SysFont("consolas", 26, bold=True)
            self._field = ce.PebbleField()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.close()
                return

        screen = self._screen
        cam = self.creature.torso.position.x - ce.WIDTH * 0.35

        def ts(p):
            return (p[0] - cam, p[1])

        # fond
        screen.fill(ce.C_BG_BOTTOM)

        # sol : relief lisse (ou plat si terrain None)
        st = 12
        pts = [(sx, ce.terrain_height(self.terrain, sx + cam))
               for sx in range(0, ce.WIDTH + st, st)]
        pygame.draw.polygon(screen, ce.C_GROUND,
                            pts + [(ce.WIDTH, ce.HEIGHT), (0, ce.HEIGHT)])
        pygame.draw.lines(screen, ce.C_GROUND_TOP, False, pts, 5)

        # cailloux : suivent le relief
        for px, py, pr, pcol in self._field.visible(cam, cam + ce.WIDTH):
            sy = py + (ce.terrain_height(self.terrain, px) - ce.GROUND_Y)
            pygame.draw.circle(screen, pcol, (px - cam, sy), pr)

        # cible (drapeau ancré au relief)
        tx = ce.TARGET_X - cam
        gyt = ce.terrain_height(self.terrain, ce.TARGET_X)
        pygame.draw.line(screen, (80, 60, 40), (tx, gyt), (tx, gyt - 120), 4)
        pygame.draw.polygon(screen, ce.C_TARGET,
                            [(tx, gyt - 120), (tx, gyt - 85), (tx + 45, gyt - 102)])

        # obstacles
        for body, shape in self.obstacles:
            verts = [ts(body.local_to_world(v)) for v in shape.get_vertices()]
            pygame.draw.polygon(screen, ce.C_OBSTACLE, verts)

        # rayons des capteurs (visualisation)
        origin = self.creature.torso.position
        for a in np.linspace(-0.2, 1.2, self.n_rays):
            end = (origin.x + math.cos(a) * 220.0, origin.y + math.sin(a) * 220.0)
            hit = self.space.segment_query_first(origin, end, 1,
                                                 pymunk.ShapeFilter(group=ce.CREATURE_GROUP))
            tip = hit.point if hit is not None else end
            pygame.draw.line(screen, (255, 120, 120), ts(origin), ts(tip), 1)

        # pattes lointaines / torse / pattes proches
        def draw_seg(body, length, radius, color):
            a = body.local_to_world((0, -length / 2))
            b = body.local_to_world((0, length / 2))
            pygame.draw.line(screen, color, ts(a), ts(b), int(radius * 2))
            for pt in (a, b):
                pygame.draw.circle(screen, color, ts(pt), int(radius))

        for leg in self.creature.legs:
            if not leg["near"]:
                draw_seg(leg["thigh"], leg["l_thigh"], 6, ce.C_LEG_FAR)
                draw_seg(leg["shin"], leg["l_shin"], 5, ce.C_LEG_FAR)
        torso_shape = list(self.creature.torso.shapes)[0]
        verts = [ts(self.creature.torso.local_to_world(v)) for v in torso_shape.get_vertices()]
        pygame.draw.polygon(screen, ce.C_TORSO, verts)
        for leg in self.creature.legs:
            if leg["near"]:
                draw_seg(leg["thigh"], leg["l_thigh"], 6, ce.C_LEG_NEAR)
                draw_seg(leg["shin"], leg["l_shin"], 5, ce.C_LEG_NEAR)

        # HUD
        dist = self._dist_to_target()
        for i, line in enumerate([f"pas: {self.steps}/{self.max_steps}",
                                  f"distance cible: {dist:6.0f}",
                                  f"obstacles: {'ON' if self.with_obstacles else 'OFF'}"]):
            screen.blit(self._font.render(line, True, ce.C_TEXT), (12, 10 + i * 20))

        # Bandeau (ex. checkpoint en cours), centré en haut
        if self.banner:
            label = self._bigfont.render(self.banner, True, (255, 255, 255))
            bx = ce.WIDTH // 2 - label.get_width() // 2
            bar = pygame.Surface((label.get_width() + 30, label.get_height() + 12),
                                 pygame.SRCALPHA)
            bar.fill((20, 30, 45, 180))
            screen.blit(bar, (bx - 15, 8))
            screen.blit(label, (bx, 14))

        self._clock.tick(ce.FPS)
        pygame.display.flip()

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None


# --------------------------------------------------------------------------
# Tests / démos en ligne de commande
# --------------------------------------------------------------------------
def random_rollout(render=False, episodes=3):
    mode = "human" if render else None
    env = QuadrupedEnv(with_obstacles=False, render_mode=mode)
    for ep in range(episodes):
        obs, _ = env.reset()
        total, done, trunc = 0.0, False, False
        while not (done or trunc):
            obs, r, done, trunc, info = env.step(env.action_space.sample())
            total += r
        cause = "cible" if info.get("reached") else ("chute" if info.get("fell") else "timeout")
        print(f"épisode {ep + 1}: récompense={total:8.1f}  pas={env.steps:4d}  fin={cause}")
    env.close()


if __name__ == "__main__":
    if "--check" in sys.argv:
        from gymnasium.utils.env_checker import check_env
        check_env(QuadrupedEnv(with_obstacles=True), skip_render_check=True)
        print("check_env : OK -- environnement conforme a l'API Gymnasium")
    elif "--render" in sys.argv:
        random_rollout(render=True, episodes=5)
    else:
        random_rollout(render=False, episodes=3)
