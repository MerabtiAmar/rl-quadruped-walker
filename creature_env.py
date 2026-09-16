"""
Environnement physique 2D — Créature quadrupède (Phase 1 : visualisation)
=========================================================================

Vue de COTE (gravité verticale) : la créature doit apprendre à marcher,
ce qui n'a de sens qu'avec gravité + équilibre.

Stack : pymunk (physique 2D) + pygame (fenêtre/rendu).

Contenu de la scène :
  - un sol
  - une créature : torse + 4 pattes (hanche + genou motorisés) = 8 moteurs
  - des obstacles statiques
  - une cible (drapeau) à atteindre

Pour l'instant la créature exécute une "démo de vie" (oscillation sinusoïdale
des moteurs) : ce N'EST PAS de l'apprentissage, juste de quoi vérifier que la
physique, les articulations et les moteurs fonctionnent. Le RL viendra ensuite
piloter `set_motor_rates()`.

Commandes :
  - Echap / fermer la fenêtre : quitter
  - R : réinitialiser la scène
  - Espace : activer/couper la démo de gait

Lancement :
  python creature_env.py                # fenêtre interactive
  python creature_env.py --headless     # test sans affichage (vérif physique)
"""

import os
import sys
import math
import random
import pymunk

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
WIDTH, HEIGHT = 1100, 600
FPS = 60
GRAVITY = (0.0, 900.0)          # +y vers le bas (repère pygame)
GROUND_Y = HEIGHT - 60          # hauteur du sol à l'écran

START_X = 250.0                 # position de départ de la créature (monde)
TARGET_X = 1900.0               # position de la cible (monde)

CREATURE_GROUP = 1              # toutes les pièces de la créature s'ignorent

# Couleurs (R, G, B)
C_BG_TOP     = (135, 180, 235)
C_BG_BOTTOM  = (205, 225, 245)
C_GROUND     = (90, 130, 80)
C_GROUND_TOP = (120, 165, 100)
C_TORSO      = (210, 90, 70)
C_LEG_NEAR   = (60, 70, 90)     # pattes proches (vives)
C_LEG_FAR    = (140, 150, 165)  # pattes lointaines (estompées)
C_OBSTACLE   = (110, 95, 130)
C_TARGET     = (235, 195, 60)
C_TEXT       = (30, 40, 55)
# Teintes de cailloux / touffes posés au sol (repères visuels de déplacement)
C_PEBBLES = [(70, 100, 65), (150, 145, 130), (95, 80, 70),
             (60, 90, 60), (175, 170, 150), (110, 130, 95)]


PEBBLE_CHUNK = 160   # largeur (monde) d'un morceau de terrain


class PebbleField:
    """
    Cailloux/touffes générés PROCÉDURALEMENT et DYNAMIQUEMENT.

    Le terrain est découpé en "chunks" de largeur PEBBLE_CHUNK. Chaque chunk
    produit ses propres cailloux via un tirage déterministe basé sur son index
    -> le décor est infini, se remplit automatiquement partout où va la
    créature (avant comme arrière), et reste reproductible.

    On ne génère que les chunks visibles (mis en cache), donc le coût reste
    constant quel que soit le déplacement.
    """

    def __init__(self, seed=7, density=(5, 11)):
        self.seed = seed
        self.density = density
        self._cache = {}

    def _chunk(self, c):
        items = self._cache.get(c)
        if items is None:
            rng = random.Random((c * 92821) ^ (self.seed * 68917))
            items = []
            for _ in range(rng.randint(*self.density)):
                x = (c + rng.random()) * PEBBLE_CHUNK
                y = GROUND_Y + rng.uniform(-3, HEIGHT - GROUND_Y - 6)
                r = rng.uniform(2.0, 6.0)
                color = rng.choice(C_PEBBLES)
                items.append((x, y, r, color))
            self._cache[c] = items
        return items

    def visible(self, left, right):
        """Itère les cailloux des chunks couvrant [left, right] (coords monde)."""
        c0 = int(math.floor(left / PEBBLE_CHUNK)) - 1
        c1 = int(math.floor(right / PEBBLE_CHUNK)) + 1
        for c in range(c0, c1 + 1):
            yield from self._chunk(c)


# --------------------------------------------------------------------------
# Construction de la créature
# --------------------------------------------------------------------------
def add_segment(space, center, length, mass, group, radius=7):
    """Crée un corps-segment vertical (longueur vers le bas)."""
    moment = pymunk.moment_for_segment(mass, (0, -length / 2), (0, length / 2), radius)
    body = pymunk.Body(mass, moment)
    body.position = center
    shape = pymunk.Segment(body, (0, -length / 2), (0, length / 2), radius)
    shape.friction = 1.2
    shape.elasticity = 0.0
    shape.filter = pymunk.ShapeFilter(group=group)
    space.add(body, shape)
    return body, shape


def add_leg(space, torso, hip_world, l_thigh, l_shin, near):
    """
    Construit une patte (cuisse + tibia) accrochée au torse à `hip_world`.
    Retourne un dict décrivant la patte et ses 2 moteurs.
    """
    # --- Cuisse : centre au milieu hanche->genou
    thigh_center = (hip_world[0], hip_world[1] + l_thigh / 2)
    thigh, thigh_shape = add_segment(space, thigh_center, l_thigh, mass=0.6,
                                     group=CREATURE_GROUP, radius=6)

    knee_world = (hip_world[0], hip_world[1] + l_thigh)

    # --- Tibia : centre au milieu genou->pied
    shin_center = (knee_world[0], knee_world[1] + l_shin / 2)
    shin, shin_shape = add_segment(space, shin_center, l_shin, mass=0.4,
                                   group=CREATURE_GROUP, radius=5)

    # --- Articulations (pivots) ancrées en coordonnées monde
    hip_pivot = pymunk.PivotJoint(torso, thigh, hip_world)
    knee_pivot = pymunk.PivotJoint(thigh, shin, knee_world)
    space.add(hip_pivot, knee_pivot)

    # --- Limites d'angle (empêche les pattes de se replier à l'infini)
    hip_limit = pymunk.RotaryLimitJoint(torso, thigh, -1.1, 1.1)
    knee_limit = pymunk.RotaryLimitJoint(thigh, shin, -0.1, 2.0)
    space.add(hip_limit, knee_limit)

    # --- Moteurs (pilotables : c'est l'espace d'action du futur agent RL)
    hip_motor = pymunk.SimpleMotor(torso, thigh, 0.0)
    knee_motor = pymunk.SimpleMotor(thigh, shin, 0.0)
    hip_motor.max_force = 1_800_000
    knee_motor.max_force = 1_200_000
    space.add(hip_motor, knee_motor)

    return {
        "thigh": thigh, "shin": shin,
        "l_thigh": l_thigh, "l_shin": l_shin,
        "hip_motor": hip_motor, "knee_motor": knee_motor,
        "near": near,
    }


class Creature:
    """Le quadrupède : torse + 4 pattes. Expose les moteurs pour le contrôle."""

    def __init__(self, space, x, y):
        # --- Torse (boîte)
        w, h = 130, 34
        mass = 8.0
        moment = pymunk.moment_for_box(mass, (w, h))
        self.torso = pymunk.Body(mass, moment)
        self.torso.position = (x, y)
        torso_shape = pymunk.Poly.create_box(self.torso, (w, h))
        torso_shape.friction = 0.8
        torso_shape.elasticity = 0.0
        torso_shape.filter = pymunk.ShapeFilter(group=CREATURE_GROUP)
        space.add(self.torso, torso_shape)
        self.torso_size = (w, h)

        # --- 4 pattes : 2 à l'arrière, 2 à l'avant.
        #     Décalées en x + colorées pour l'effet "perspective" (proche/loin).
        hip_y = y + h / 2
        l_thigh, l_shin = 20, 20
        offsets = [
            (-50, True),   # arrière proche
            (-62, False),  # arrière loin
            (+50, True),   # avant proche
            (+62, False),  # avant loin
        ]
        self.legs = []
        for dx, near in offsets:
            leg = add_leg(space, self.torso, (x + dx, hip_y), l_thigh, l_shin, near)
            self.legs.append(leg)

    def set_motor_rates(self, rates):
        """
        rates : liste de 8 vitesses angulaires
                [hanche0, genou0, hanche1, genou1, ...].
        --> Interface que l'agent RL utilisera plus tard.
        """
        for i, leg in enumerate(self.legs):
            leg["hip_motor"].rate = rates[2 * i]
            leg["knee_motor"].rate = rates[2 * i + 1]

    def demo_gait(self, t):
        """Démo non-apprenante : oscillation pour vérifier la mécanique."""
        rates = []
        for i, leg in enumerate(self.legs):
            phase = math.pi if (i >= 2) else 0.0   # avant/arrière en opposition
            hip = 3.0 * math.sin(2.0 * math.pi * 0.8 * t + phase)
            knee = 2.5 * math.sin(2.0 * math.pi * 0.8 * t + phase + math.pi / 2)
            rates += [hip, knee]
        self.set_motor_rates(rates)

    @property
    def position(self):
        return self.torso.position


# --------------------------------------------------------------------------
# Construction du monde
# --------------------------------------------------------------------------
def generate_obstacles(rng, difficulty=1.0):
    """
    Génère une disposition d'obstacles ALÉATOIRE (domain randomization).

    À chaque appel, le nombre, les positions, les largeurs, les hauteurs et les
    espacements varient -> la créature doit apprendre une compétence GÉNÉRALE
    (franchir un obstacle) plutôt que de mémoriser une scène fixe.

    `difficulty` ∈ [0,1] règle la sévérité :
      - hauteurs : restent FRANCHISSABLES (plafonnées par rapport aux pattes)
      - espacements : se resserrent quand la difficulté monte
    """
    difficulty = max(0.0, min(1.0, difficulty))
    leg_reach = 40.0                      # portée des pattes (l_thigh + l_shin)
    h_min = 10.0
    h_max = leg_reach * (0.6 + 1.3 * difficulty)   # ~36 px (facile) -> ~76 px (dur)

    specs = []
    x = START_X + rng.uniform(260, 410)            # 1er obstacle pas trop proche
    x_end = TARGET_X - 220
    while x < x_end:
        w = rng.uniform(25, 60)
        h = rng.uniform(h_min, max(h_min + 6, h_max))   # forte variété de hauteur
        specs.append((x, w, h))
        gap = rng.uniform(160 - 60 * difficulty, 340 - 150 * difficulty)
        x += w + max(110.0, gap)                   # espacements plus serrés
    return specs


def generate_terrain(rng, difficulty=1.0, rough=True):
    """
    Génère un relief LISSE et CONTINU : une somme de quelques sinusoïdes de
    fréquences/phases aléatoires (rolling hills). Renvoie la liste des
    composantes (amplitude, fréquence, phase), ou None pour un sol plat.

    `difficulty` règle l'amplitude (plus c'est dur, plus le relief est marqué).
    Les pentes restent modérées pour rester franchissables.
    """
    if not rough:
        return None
    difficulty = max(0.0, min(1.0, difficulty))
    base = 22.0 + 48.0 * difficulty                 # amplitude totale ~22 -> ~70 px
    comps = []
    for fmin, fmax, frac in [(0.0018, 0.0034, 0.50),   # longues ondulations
                             (0.0045, 0.0085, 0.34),   # collines moyennes
                             (0.0100, 0.0170, 0.16)]:   # petits reliefs (plus marqués)
        comps.append((base * frac, rng.uniform(fmin, fmax),
                      rng.uniform(0, 2 * math.pi)))
    return comps


def terrain_height(terrain, x):
    """Hauteur (y écran) du sol à l'abscisse monde `x`. GROUND_Y si plat."""
    if not terrain:
        return GROUND_Y
    y = GROUND_Y
    for amp, freq, phase in terrain:
        y -= amp * math.sin(freq * x + phase)
    return y


def build_world(with_obstacles=True, spawn_clearance=78, rng=None,
                difficulty=1.0, rough_terrain=False):
    """
    Construit l'espace physique.

    with_obstacles  : pose (ou non) les obstacles -> permet le curriculum.
    spawn_clearance : hauteur d'apparition du torse AU-DESSUS du sol local.
    rng             : random.Random pour la randomisation (obstacles + relief).
    difficulty      : sévérité des obstacles et du relief.
    rough_terrain   : True -> terrain ondulé ; False -> sol plat.
    """
    if rng is None:
        rng = random.Random()

    space = pymunk.Space()
    space.gravity = GRAVITY

    terrain = generate_terrain(rng, difficulty, rough=rough_terrain)

    # --- Sol : segment plat unique, ou chaîne de segments suivant le relief
    if terrain is None:
        ground = pymunk.Segment(space.static_body, (-2000, GROUND_Y),
                                (4000, GROUND_Y), 5)
        ground.friction = 1.0
        ground.elasticity = 0.0
        space.add(ground)
    else:
        step = 16
        prev = (-2000, terrain_height(terrain, -2000))
        for x in range(-2000 + step, 4000 + step, step):
            cur = (x, terrain_height(terrain, x))
            seg = pymunk.Segment(space.static_body, prev, cur, 5)
            seg.friction = 1.0
            seg.elasticity = 0.0
            space.add(seg)
            prev = cur

    # --- Obstacles randomisés posés SUR le relief
    obstacles = []
    if with_obstacles:
        for ox, ow, oh in generate_obstacles(rng, difficulty):
            gy = terrain_height(terrain, ox)
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = (ox, gy - oh / 2)
            shape = pymunk.Poly.create_box(body, (ow, oh))
            shape.friction = 1.0
            shape.elasticity = 0.0
            space.add(body, shape)
            obstacles.append((body, shape))

    # --- Créature posée au-dessus du sol local, à START_X
    spawn_y = terrain_height(terrain, START_X) - spawn_clearance
    creature = Creature(space, START_X, spawn_y)
    return space, creature, obstacles, terrain


# --------------------------------------------------------------------------
# Rendu (caméra qui suit la créature horizontalement)
# --------------------------------------------------------------------------
def run_gui():
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Créature quadrupède — environnement physique 2D")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    big = pygame.font.SysFont("consolas", 22, bold=True)

    space, creature, obstacles, terrain = build_world()
    field = PebbleField()
    gait_on = True
    t = 0.0

    # Dégradé de fond pré-rendu
    bg = pygame.Surface((WIDTH, HEIGHT))
    for y in range(HEIGHT):
        f = y / HEIGHT
        col = [int(C_BG_TOP[i] * (1 - f) + C_BG_BOTTOM[i] * f) for i in range(3)]
        pygame.draw.line(bg, col, (0, y), (WIDTH, y))

    def cam_x():
        return creature.position.x - WIDTH * 0.35

    def to_screen(p):
        return (p[0] - cam_x(), p[1])

    def draw_segment_body(body, length, radius, color):
        a = body.local_to_world((0, -length / 2))
        b = body.local_to_world((0, length / 2))
        pygame.draw.line(screen, color, to_screen(a), to_screen(b), int(radius * 2))
        for pt in (a, b):
            pygame.draw.circle(screen, color, to_screen(pt), int(radius))

    def draw_poly(body, shape, color, outline=None):
        verts = [to_screen(body.local_to_world(v)) for v in shape.get_vertices()]
        pygame.draw.polygon(screen, color, verts)
        if outline:
            pygame.draw.polygon(screen, outline, verts, 2)

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE,):
                    running = False
                elif e.key == pygame.K_r:
                    space, creature, obstacles, terrain = build_world()
                    t = 0.0
                elif e.key == pygame.K_SPACE:
                    gait_on = not gait_on

        # --- Physique
        dt = 1.0 / FPS
        if gait_on:
            creature.demo_gait(t)
        else:
            creature.set_motor_rates([0.0] * 8)
        substeps = 6
        for _ in range(substeps):
            space.step(dt / substeps)
        t += dt

        # --- Rendu
        screen.blit(bg, (0, 0))

        # Sol
        gy = GROUND_Y
        pygame.draw.rect(screen, C_GROUND, (0, gy, WIDTH, HEIGHT - gy))
        pygame.draw.rect(screen, C_GROUND_TOP, (0, gy - 5, WIDTH, 6))

        # Cailloux / touffes : générés dynamiquement pour la portion visible
        left = cam_x()
        for px, py, pr, pcol in field.visible(left, left + WIDTH):
            pygame.draw.circle(screen, pcol, (px - left, py), pr)

        # Cible (drapeau)
        tx = TARGET_X - cam_x()
        if -50 < tx < WIDTH + 50:
            pygame.draw.line(screen, (80, 60, 40), (tx, gy), (tx, gy - 120), 4)
            pygame.draw.polygon(screen, C_TARGET,
                                [(tx, gy - 120), (tx, gy - 85), (tx + 45, gy - 102)])
        # Zone-cible translucide
        zone = pygame.Surface((60, HEIGHT), pygame.SRCALPHA)
        zone.fill((*C_TARGET, 50))
        screen.blit(zone, (tx - 30, 0))

        # Obstacles
        for body, shape in obstacles:
            draw_poly(body, shape, C_OBSTACLE, outline=(70, 55, 90))

        # Pattes lointaines (derrière le torse)
        for leg in creature.legs:
            if not leg["near"]:
                draw_segment_body(leg["thigh"], leg["l_thigh"], 6, C_LEG_FAR)
                draw_segment_body(leg["shin"], leg["l_shin"], 5, C_LEG_FAR)

        # Torse
        torso_shape = list(creature.torso.shapes)[0]
        draw_poly(creature.torso, torso_shape, C_TORSO, outline=(150, 50, 40))
        # petit "oeil" pour donner un sens de direction
        head = creature.torso.local_to_world((creature.torso_size[0] / 2 - 8, -4))
        pygame.draw.circle(screen, (255, 255, 255), to_screen(head), 6)
        pygame.draw.circle(screen, (20, 20, 20), to_screen(head), 3)

        # Pattes proches (devant le torse)
        for leg in creature.legs:
            if leg["near"]:
                draw_segment_body(leg["thigh"], leg["l_thigh"], 6, C_LEG_NEAR)
                draw_segment_body(leg["shin"], leg["l_shin"], 5, C_LEG_NEAR)

        # HUD
        dist = TARGET_X - creature.position.x
        hud = [
            f"Distance cible : {dist:7.0f}",
            f"Gait demo : {'ON' if gait_on else 'OFF'}  (Espace)",
            "R : reset    Echap : quitter",
        ]
        for i, line in enumerate(hud):
            screen.blit(font.render(line, True, C_TEXT), (15, 12 + i * 22))
        if dist < 60:
            msg = big.render("CIBLE ATTEINTE !", True, (200, 120, 0))
            screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, 30))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


# --------------------------------------------------------------------------
# Mode headless : vérifie que la physique tourne sans crash / sans diverger
# --------------------------------------------------------------------------
def run_headless(steps=600):
    space, creature, _ = build_world()
    t = 0.0
    dt = 1.0 / FPS
    for _ in range(steps):
        creature.demo_gait(t)
        for _ in range(6):
            space.step(dt / 6)
        t += dt
    p = creature.position
    ok = math.isfinite(p.x) and math.isfinite(p.y) and p.y < GROUND_Y + 50
    print(f"[headless] {steps} pas simulés.")
    print(f"[headless] position torse finale : x={p.x:.1f}, y={p.y:.1f}")
    print(f"[headless] physique stable : {'OUI' if ok else 'NON (divergence !)'}")
    return ok


if __name__ == "__main__":
    if "--headless" in sys.argv:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        ok = run_headless()
        sys.exit(0 if ok else 1)
    else:
        run_gui()
