import math

import numpy as np
import pymunk

from .base import DT, REC_EVERY, SUBSTEPS, TAU, Env, clamp, rounded, u01

X0, H0 = 0.6, 1.7
G = 9.81
HAND_W, HAND_H = 0.3, 0.03
HAND_X = (X0 - 1.5, X0 + 1.5)
HAND_Y = (H0 - 0.06 - 0.5, H0 - 0.06 + 0.2)
MAX_T = 4.5


class Pizza(Env):
    key = "pizza"
    name = "Pizza chef"
    lede = ("The chef tosses the dough and tries to catch it. At the toss the network picks the throw power, "
            "the spin, how wobbly the release is and the drift. While the dough flies it moves the chef's "
            "hands. The ceiling height and a draught change every episode.")
    hint = "A quiet pat is a clean catch, a loud splat is a mess"
    stat_label = "Pizza"
    vars = [
        ("size", "pizza width at the end, cm"),
        ("roundness", "1 if it flew flat, lower the more it tumbled"),
        ("height", "top of the toss, m"),
        ("spins", "full turns in the air"),
        ("caught", "1 if it ends up on the chef's hands, else 0"),
        ("dropped", "1 if it lands on the floor, else 0"),
        ("torn", "1 if it stretched until it tore, else 0"),
        ("ceiling", "1 if it hit the ceiling, else 0"),
        ("reach", "how far the hands moved from the start, m"),
    ]
    presets = [
        ("Good pizza", "size*roundness - 40*dropped - 40*torn - 40*ceiling"),
        ("Biggest pizza", "size"),
        ("Perfect circle", "100*roundness - 40*dropped"),
        ("High toss", "20*height - 40*dropped"),
        ("Make a mess", "dropped + torn + ceiling"),
    ]
    obs_names = ["toss phase", "ceiling height", "draught", "dough dx", "dough dy", "dough vx", "dough vy",
                 "sin tilt", "cos tilt", "tilt rate", "hands x", "hands y", "time"]
    act_names = ["power / hands x", "spin / hands y", "wobble", "drift"]

    def __init__(self):
        s = self.space = pymunk.Space()
        s.gravity = (0, -G)
        s.iterations = 12
        floor = pymunk.Segment(s.static_body, (-20, 0), (20, 0), 0.0)
        floor.friction, floor.elasticity = 0.8, 0.02
        self.hands = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        hs = pymunk.Poly.create_box(self.hands, (2 * HAND_W, 2 * HAND_H))
        hs.friction, hs.elasticity = 1.0, 0.02
        self.hand_shape = hs
        s.add(floor, self.hands, hs)
        self.dough = None

    def _make_dough(self, r):
        if self.dough is not None:
            self.space.remove(self.dough, self.dough_shape)
        b = pymunk.Body(0.6, pymunk.moment_for_box(0.6, (2 * r, 0.04)))
        sh = pymunk.Poly.create_box(b, (2 * r, 0.04))
        sh.friction, sh.elasticity = 0.8, 1.0
        self.space.add(b, sh)
        self.dough, self.dough_shape = b, sh

    def reset(self, rng):
        self.ceil = float(rng.uniform(3.0, 4.3))
        self.wind = float(rng.uniform(-0.8, 0.8))
        self.hands.position = (X0, H0 - 0.06)
        self.hands.velocity = (0, 0)
        self._make_dough(0.12)
        self.dough.position = (X0, H0 + 0.05)
        self.phase = 0
        self.k = 0
        self.t = 0.0
        self.om = 0.0
        self.r = 0.12
        self.tilt = 0.0
        self.peak = H0
        self.hit = -1.0
        self.stuck = False
        self.t_floor = -1.0
        self.frames = []
        return self._obs()

    def _obs(self):
        d, h = self.dough, self.hands
        return np.array([
            1.0 if self.phase == 0 else 0.0, (self.ceil - 3.6) / 0.6, self.wind / 0.8,
            d.position.x - h.position.x, d.position.y - h.position.y, d.velocity.x / 3, d.velocity.y / 5,
            math.sin(d.angle), math.cos(d.angle), d.angular_velocity / 5,
            h.position.x - X0, h.position.y - (H0 - 0.06), self.t / 2,
        ], dtype=np.float32)

    def step(self, a):
        if self.phase == 0:
            g = [u01(x) for x in a]
            vy, self.om, wob, vx = 2 + 6.5 * g[0], 2 + 20 * g[1], g[2], (g[3] - 0.5) * 2.4
            top = self.ceil - 0.03 - H0
            tau = (vy - math.sqrt(vy * vy - 2 * G * top)) / G if vy * vy / (2 * G) > top else 2 * vy / G
            self.r = 0.12 + 0.014 * self.om * tau
            self._make_dough(self.r)
            self.dough.position = (X0, H0 + 0.05)
            self.dough.velocity = (vx, vy)
            self.dough.angular_velocity = 6 * wob
            self.phase = 1
            return self._obs(), False
        if self.hit < 0:
            h = self.hands
            vx, vy = 2.0 * float(a[0]), 1.0 * float(a[1])
            x, y = h.position
            if (x <= HAND_X[0] and vx < 0) or (x >= HAND_X[1] and vx > 0):
                vx = 0.0
            if (y <= HAND_Y[0] and vy < 0) or (y >= HAND_Y[1] and vy > 0):
                vy = 0.0
            h.velocity = (vx, vy)
        for _ in range(SUBSTEPS):
            if self._substep():
                return self._obs(), True
        if self.hit >= 0:
            while not self._substep():
                pass
            return self._obs(), True
        return self._obs(), False

    def _touching(self):
        hit = False

        def look(arb):
            nonlocal hit
            hit = True
        self.dough.each_arbiter(look)
        return hit

    def _substep(self):
        d, h = self.dough, self.hands
        if self.k % REC_EVERY == 0:
            self.frames.extend((d.position.x, d.position.y, d.angle, self.om * self.t, h.position.x, h.position.y))
        if self.stuck:
            self.k += 1
            self.t = self.k * DT
            return self.t > self.hit + 1.0
        if self.hit < 0:
            d.force = (0.6 * self.wind, 0)
        else:
            h.velocity = (0, 0)
        self.space.step(DT)
        self.k += 1
        self.t = self.k * DT
        x, y = h.position
        h.position = (clamp(x, *HAND_X), clamp(y, *HAND_Y))
        if self.hit < 0:
            self.tilt = max(self.tilt, math.acos(clamp(math.cos(d.angle), -1, 1)))
            if self.t > 0.08 and self._touching():
                self.hit = self.t
        else:
            d.velocity = d.velocity * (0.5 ** DT)
            d.angular_velocity *= 0.1 ** DT
        self.peak = max(self.peak, d.position.y)
        if not self.stuck and d.position.y >= self.ceil - 0.05:
            self.stuck = True
            self.hit = self.t
        if self.t_floor < 0 and d.position.y < 0.1:
            self.t_floor = self.t
        if self.hit >= 0 and self.t > self.hit + 0.8:
            return True
        return self.t >= MAX_T

    def _on_hands(self):
        d, h = self.dough, self.hands
        return abs(d.position.x - h.position.x) < HAND_W + 0.15 and 0 < d.position.y - h.position.y < 0.4

    def metrics(self):
        d = self.dough
        ceiling = 1.0 if self.stuck else 0.0
        dropped = 1.0 if not self.stuck and d.position.y < H0 - 0.4 else 0.0
        air = self.hit if self.hit >= 0 else self.t
        return {
            "size": 200 * self.r, "roundness": max(0.0, 1 - self.tilt / 1.2), "height": self.peak,
            "spins": self.om * air / TAU, "caught": 1.0 if not self.stuck and not dropped and self._on_hands() else 0.0,
            "dropped": dropped, "torn": 1.0 if self.r > 0.4 else 0.0, "ceiling": ceiling,
            "reach": math.hypot(self.hands.position.x - X0, self.hands.position.y - (H0 - 0.06)),
        }

    def replay(self):
        m = self.metrics()
        if m["dropped"] and self.t_floor >= 0:
            E = self.t_floor
        elif self.hit >= 0:
            E = self.hit
        else:
            E = self.t
        n = len(self.frames) // 6
        i = min(n - 1, round(E * 60))
        return {"fr": rounded(self.frames), "r": round(self.r, 4), "ceil": round(self.ceil, 3), "wind": round(self.wind, 3),
                "air": round(self.hit if self.hit >= 0 else self.t, 3), "xE": self.frames[i * 6] if n else X0,
                "E": round(E, 3), "T": round(E + 1.0, 3)}
