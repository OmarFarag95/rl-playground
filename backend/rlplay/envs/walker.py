import math

import numpy as np
import pymunk

from .base import REC_EVERY, Env, clamp, rounded

DT = 1 / 120
SUBSTEPS = 6                      # 20 decisions a second
# Body: torso, then for each leg a thigh and a shin with the foot built in. Sizes in metres, masses in kg.
TORSO = (0.34, 0.62, 30.0)        # width, height, mass
THIGH = (0.13, 0.46, 7.0)
SHIN = (0.11, 0.46, 4.0)
FOOT = (0.28, 0.07)               # foot under the shin, sticking forward
HIP_Y = 0.46 + 0.46 + 0.07        # hip height when standing straight
# Joint targets the network can ask for (radians, relative to the parent part)
HIP_RANGE = (-0.9, 1.1)           # + swings the leg forward
KNEE_RANGE = (-1.9, 0.0)          # - bends the knee
HIP_K, HIP_D = 1100.0, 70.0       # motor stiffness and damping
KNEE_K, KNEE_D = 800.0, 45.0
N_RAYS = 10
RAY_LEN = 3.2
GROUP = 1                         # the walker's parts never collide with each other


def hip_target(a):
    """0 = leg straight down, +1 = swung forward, -1 = swung back."""
    return float(a) * (HIP_RANGE[1] if a > 0 else -HIP_RANGE[0])


def knee_target(a):
    """0 or more = straight knee, -1 = fully bent."""
    return min(0.0, float(a)) * -KNEE_RANGE[0]


def _box(body, w, h, offset=(0, 0)):
    x, y = offset
    return pymunk.Poly(body, [(x - w / 2, y - h / 2), (x + w / 2, y - h / 2), (x + w / 2, y + h / 2), (x - w / 2, y + h / 2)])


class Walker(Env):
    key = "walker"
    name = "Walker"
    dense = True                  # the formula is scored at every step, not only at the end
    hp = {"minibatch": 512, "epochs": 6}  # long episodes: bigger minibatches keep updates quick
    live = True                   # supports live play with pokes from the browser
    parallel = True               # episodes are heavy: step them in worker processes
    lede = ("A two-legged ragdoll learns to walk over rough ground. Twenty times a second the network sees its "
            "body, its feet and ten lidar rays reaching into the ground ahead, and sets a target angle for each hip "
            "and knee. Then watch it live and get in its way: shove it, drop crates in front of it, turn on the wind.")
    hint = "A thud is a fall"
    stat_label = "Distance"
    vars = [
        ("distance", "how far the torso has moved forward, m"),
        ("fell", "1 if it fell over, else 0"),
        ("time", "seconds on its feet"),
        ("speed", "average forward speed, m/s"),
        ("effort", "how hard the motors pushed, summed over time"),
        ("wobble", "how much the torso tilted, summed over time"),
        ("hops", "seconds with both feet off the ground"),
        ("steps", "foot landings"),
    ]
    presets = [
        ("Walk forward", "10*distance - 20*fell"),
        ("Walk smoothly", "10*distance - 20*fell - effort"),
        ("Sprint", "20*distance - 20*fell"),
        ("Just stay up", "time - 20*fell"),
        ("Graceful", "10*distance - 20*fell - 5*wobble"),
        ("Moonwalk", "-10*distance - 20*fell"),
        ("Bunny hop", "10*distance + 5*hops - 20*fell"),
    ]
    obs_names = (["sin tilt", "cos tilt", "spin rate", "speed x", "speed y", "height",
                  "L hip", "L knee", "R hip", "R knee", "L hip speed", "L knee speed", "R hip speed", "R knee speed",
                  "L foot down", "R foot down"] + [f"lidar {i + 1}" for i in range(N_RAYS)])
    act_names = ["L hip", "L knee", "R hip", "R knee"]
    settings = [
        ("rough", "Rough ground", "Height of the bumps in the ground, m", 0.0, 0.3, 0.01, 0.08),
        ("crates", "Crates", "Crates lying on the course each episode", 0, 8, 1, 0),
        ("shoves", "Random shoves", "Pushes from nowhere per 10 seconds, to make it robust", 0, 10, 1, 0),
        ("wind", "Wind gusts", "Strongest gust, N (blows either way, changes every few seconds)", 0, 250, 5, 0),
        ("strength", "Muscle strength", "Motor strength, × normal", 0.5, 1.5, 0.05, 1.0),
        ("friction", "Grip", "Friction between feet and ground (0.2 is ice)", 0.2, 1.5, 0.05, 1.0),
        ("gravity", "Gravity", "9.81 on Earth, 1.62 on the Moon, 3.71 on Mars, m/s²", 1.6, 20.0, 0.01, 9.81),
        ("max_time", "Episode length", "Seconds before the episode ends, if it has not fallen", 4, 30, 1, 12),
    ]
    setting_presets = [
        ("Flat and calm", {"rough": 0}),
        ("Obstacle course", {"rough": 0.12, "crates": 5}),
        ("Storm", {"wind": 200, "shoves": 4}),
        ("Ice rink", {"rough": 0, "friction": 0.25}),
        ("Moon", {"gravity": 1.62}),
    ]

    # ----- building ----------------------------------------------------------
    def _build(self):
        s = self.space = pymunk.Space()
        s.iterations = 15
        filt = pymunk.ShapeFilter(group=GROUP)
        self.parts = []          # torso, L thigh, L shin, R thigh, R shin
        tw, th, tm = TORSO
        torso = pymunk.Body(tm, pymunk.moment_for_box(tm, (tw, th)))
        ts = _box(torso, tw, th)
        ts.filter, ts.friction = filt, 0.8
        s.add(torso, ts)
        self.parts.append(torso)
        self.torso_shape = ts
        self.feet = []
        self.motors = []
        for _ in range(2):
            thigh = pymunk.Body(THIGH[2], pymunk.moment_for_box(THIGH[2], THIGH[:2]))
            sh1 = _box(thigh, *THIGH[:2])
            shin = pymunk.Body(SHIN[2], pymunk.moment_for_box(SHIN[2], SHIN[:2]))
            sh2 = _box(shin, *SHIN[:2])
            foot = _box(shin, FOOT[0], FOOT[1], (0.07, -SHIN[1] / 2 - FOOT[1] / 2))
            for sh in (sh1, sh2, foot):
                sh.filter = filt
                sh.friction = 1.0
            s.add(thigh, sh1, shin, sh2, foot)
            hip = pymunk.PivotJoint(torso, thigh, (0, -th / 2), (0, THIGH[1] / 2))
            knee = pymunk.PivotJoint(thigh, shin, (0, -THIGH[1] / 2), (0, SHIN[1] / 2))
            hip_lim = pymunk.RotaryLimitJoint(torso, thigh, HIP_RANGE[0] - 0.15, HIP_RANGE[1] + 0.15)
            knee_lim = pymunk.RotaryLimitJoint(thigh, shin, KNEE_RANGE[0] - 0.1, KNEE_RANGE[1] + 0.05)
            hip_m = pymunk.DampedRotarySpring(torso, thigh, 0.0, HIP_K, HIP_D)
            knee_m = pymunk.DampedRotarySpring(thigh, shin, 0.0, KNEE_K, KNEE_D)
            s.add(hip, knee, hip_lim, knee_lim, hip_m, knee_m)
            self.parts += [thigh, shin]
            self.feet.append(foot)
            self.motors += [hip_m, knee_m]
        self.ground = []
        self.crates = []

    def _terrain(self, rng):
        for g in self.ground:
            self.space.remove(g)
        rough = self.cfg["rough"]
        xs = np.arange(-60.0, 120.0, 0.8)  # room behind too, for walking backwards
        h = np.zeros(len(xs))
        start = int(np.searchsorted(xs, 0.0))
        for order in (range(start, len(xs)), range(start - 1, -1, -1)):
            y = 0.0
            for i in order:
                if abs(xs[i]) > 2.0:  # a flat start, then a gentle random walk with bumps
                    y = clamp(y + rng.normal(0, rough * 0.6), -1.5, 1.5) * 0.97
                    h[i] = y + rng.uniform(-rough, rough) * 0.5
        self.tx, self.ty = xs, h
        body = self.space.static_body
        self.ground = []
        for i in range(len(xs) - 1):
            seg = pymunk.Segment(body, (xs[i], h[i]), (xs[i + 1], h[i + 1]), 0.02)
            seg.friction = self.cfg["friction"]
            self.ground.append(seg)
        self.space.add(*self.ground)

    def ground_y(self, x):
        return float(np.interp(x, self.tx, self.ty))

    def add_crate(self, x, size=0.4, drop=0.0):
        """A loose crate at ``x``, resting on the ground or dropped from ``drop`` metres up."""
        m = 12.0 * (size / 0.4) ** 2
        b = pymunk.Body(m, pymunk.moment_for_box(m, (size, size)))
        b.position = (x, self.ground_y(x) + size / 2 + 0.02 + drop)
        sh = _box(b, size, size)
        sh.friction = 0.9
        self.space.add(b, sh)
        self.crates.append({"body": b, "shape": sh, "size": size, "born": len(self.frames) // self.FR})
        return b

    # ----- episode -----------------------------------------------------------
    FR = 15  # floats per replay frame: (x, y, angle) for the five parts

    def reset(self, rng):
        if not hasattr(self, "space"):
            self._build()
        self.rng = rng
        s = self.space
        for c in self.crates:
            s.remove(c["body"], c["shape"])
        self.crates = []
        self.frames = []
        self.crate_frames = []
        s.gravity = (0, -self.cfg["gravity"])
        self._terrain(rng)
        strength = self.cfg["strength"]
        for k, m in enumerate(self.motors):
            hip = k % 2 == 0
            m.stiffness = (HIP_K if hip else KNEE_K) * strength
            m.damping = HIP_D if hip else KNEE_D
            m.rest_angle = 0.0
        x0, y0 = 0.0, self.ground_y(0.0) + HIP_Y + 0.01
        tw, th, _ = TORSO
        place = [(x0, y0 + th / 2)]
        for _ in range(2):
            place += [(x0, y0 - THIGH[1] / 2), (x0, y0 - THIGH[1] - SHIN[1] / 2)]
        for b, p in zip(self.parts, place):
            b.position = p
            b.velocity = (0, 0)
            b.angle = 0.0
            b.angular_velocity = 0.0
        self.events = []
        for _ in range(int(self.cfg["crates"])):
            self.add_crate(float(rng.uniform(3.0, 40.0)), float(rng.uniform(0.25, 0.5)))
        self.k = 0
        self.t = 0.0
        self.x0 = x0
        self.fell = False
        self.effort = 0.0
        self.wobble = 0.0
        self.hops = 0.0
        self.steps = 0
        self.down = [True, True]
        self.wind = 0.0
        self.wind_next = 0.0
        self.shove_rate = self.cfg["shoves"] / 10.0
        self.winds = []          # (time, force) whenever the wind changes
        self.last_a = np.zeros(4)
        self.done = False
        self.final = None
        return self._obs()

    def _contacts(self):
        down = [False, False]
        for i, f in enumerate(self.feet):
            def look(arb, i=i):
                down[i] = True
            f.body.each_arbiter(look)
        return down

    def _obs(self):
        tb = self.parts[0]
        gy = self.ground_y(tb.position.x)
        rel = []
        spd = []
        for leg in range(2):
            thigh, shin = self.parts[1 + 2 * leg], self.parts[2 + 2 * leg]
            rel += [thigh.angle - tb.angle, shin.angle - thigh.angle]
            spd += [thigh.angular_velocity - tb.angular_velocity, shin.angular_velocity - thigh.angular_velocity]
        down = self._contacts()
        rays = []
        hip = tb.local_to_world((0, -TORSO[1] / 2))
        filt = pymunk.ShapeFilter(group=GROUP)  # rays pass through the walker itself
        for i in range(N_RAYS):
            ang = -math.pi / 2 + (i + 0.5) / N_RAYS * (math.pi / 2 * 0.95)  # straight down to nearly level, ahead
            end = (hip.x + RAY_LEN * math.cos(ang), hip.y + RAY_LEN * math.sin(ang))
            q = self.space.segment_query_first(hip, end, 0.0, filt)
            rays.append(q.alpha if q else 1.0)
        return np.array([
            math.sin(tb.angle), math.cos(tb.angle), tb.angular_velocity / 5, tb.velocity.x / 3, tb.velocity.y / 3,
            (tb.position.y - gy - (HIP_Y + TORSO[1] / 2)) / 0.5,
            *[v / 1.5 for v in rel], *[v / 10 for v in spd], float(down[0]), float(down[1]), *rays,
        ], dtype=np.float32)

    def step(self, a):
        targets = [hip_target(a[0]), knee_target(a[1]), hip_target(a[2]), knee_target(a[3])]
        for m, tgt in zip(self.motors, targets):
            m.rest_angle = tgt
        # effort: jerky changes plus holding force; a steady 12 s walk costs a few points
        self.effort += float(np.abs(np.asarray(a) - self.last_a).sum()) * 0.02 + 0.005 * float(np.abs(a).sum())
        self.last_a = np.asarray(a, dtype=float)
        self._random_trouble()
        for _ in range(SUBSTEPS):
            self._substep()
            if self.fell:
                break
        if self.fell or self.t >= self.cfg["max_time"]:
            self.done = True
            self.final = self.metrics()
            if self.fell:
                self._collapse()
        return self._obs(), self.done

    def _collapse(self, secs=0.8):
        """After a fall the muscles let go, so the replay shows the body hitting the ground."""
        for m in self.motors:
            m.stiffness *= 0.04
            m.damping *= 0.3
        for _ in range(int(secs / DT)):
            if self.k % REC_EVERY == 0:
                for b in self.parts:
                    self.frames.extend((b.position.x, b.position.y, b.angle))
                self.crate_frames.append([v for c in self.crates for v in (c["body"].position.x, c["body"].position.y, c["body"].angle)])
            self.space.step(DT)
            self.k += 1

    def _random_trouble(self):
        """Shoves and gusts from the settings, so a network can learn to cope with them."""
        rng = self.rng
        if self.cfg["wind"] > 0 and self.t >= self.wind_next:
            self.set_wind(float(rng.uniform(-1, 1)) * self.cfg["wind"])
            self.wind_next = self.t + float(rng.uniform(2.0, 5.0))
        if self.shove_rate > 0 and rng.random() < self.shove_rate * SUBSTEPS * DT:
            self.shove(float(rng.choice([-1, 1])) * float(rng.uniform(80, 160)))

    def _substep(self):
        if self.k % REC_EVERY == 0:
            for b in self.parts:
                self.frames.extend((b.position.x, b.position.y, b.angle))
            self.crate_frames.append([v for c in self.crates for v in (c["body"].position.x, c["body"].position.y, c["body"].angle)])
        if self.wind:
            share = self.wind / len(self.parts)
            for b in self.parts:
                b.apply_force_at_world_point((share, 0), b.position)
        self.space.step(DT)
        self.k += 1
        self.t = self.k * DT
        tb = self.parts[0]
        self.wobble += abs(tb.angle) * DT
        down = self._contacts()
        if not down[0] and not down[1]:
            self.hops += DT
        for i in range(2):
            if down[i] and not self.down[i]:
                self.steps += 1
        self.down = down
        # a fall: the torso tips too far, or touches the ground
        touching = []
        tb.each_arbiter(lambda arb: touching.append(1))
        if abs(tb.angle) > 1.1 or touching or tb.position.y - self.ground_y(tb.position.x) < 0.75:
            self.fell = True

    # ----- pokes from the browser (and from the random-trouble settings) --------
    def shove(self, impulse):
        """A sideways shove to the torso, in N·s (negative pushes it backwards)."""
        tb = self.parts[0]
        tb.apply_impulse_at_world_point((impulse, 0.15 * abs(impulse)), tb.local_to_world((0, 0.15)))
        self.events.append({"t": round(self.t, 3), "k": "shove", "v": round(impulse, 1)})

    def set_wind(self, force):
        self.wind = float(force)
        self.winds.append([round(self.t, 3), round(self.wind, 1)])

    def poke(self, kind, value=0.0, x=None):
        """Interference from the live view."""
        if self.done:
            return
        if kind == "shove":
            self.shove(clamp(float(value), -400, 400))
        elif kind == "wind":
            self.set_wind(clamp(float(value), -400, 400))
        elif kind == "crate":
            x = float(self.parts[0].position.x + 1.6) if x is None else float(x)
            self.add_crate(x, clamp(float(value) or 0.4, 0.2, 0.8), drop=2.5)
            self.events.append({"t": round(self.t, 3), "k": "crate", "x": round(x, 2)})

    # ----- results -----------------------------------------------------------
    def metrics(self):
        if self.final is not None:   # frozen at the moment the episode ended
            return dict(self.final)
        tb = self.parts[0]
        dist = tb.position.x - self.x0
        time_up = self.t
        return {"distance": dist, "fell": 1.0 if self.fell else 0.0, "time": time_up,
                "speed": dist / max(time_up, 1e-6), "effort": self.effort, "wobble": self.wobble,
                "hops": self.hops, "steps": float(self.steps)}

    def terrain(self):
        return rounded(np.stack([self.tx, self.ty], 1).ravel())

    def replay(self):
        n = len(self.frames) // self.FR
        end = round(self.final["time"] if self.final else n / 60, 3)
        return {"fr": rounded(self.frames), "terrain": self.terrain(),
                "crates": [{"size": c["size"], "born": c["born"]} for c in self.crates],
                "cfr": [rounded(f) for f in self.crate_frames], "winds": self.winds, "events": self.events,
                "E": end, "T": round(n / 60 + 0.6, 3)}
