import math

import numpy as np
import pymunk

from .base import DT, REC_EVERY, SUBSTEPS, TAU, Env, rounded, u01

HX, HY = 0.45, 0.37
X0 = 0.6
MOUTH = (0.7, 0.22)
MAX_T = 6.5


class Dog(Env):
    key = "dog"
    name = "Frisbee dog"
    lede = ("Every episode the frisbee is thrown differently. The network watches the dog and the frisbee "
            "20 times a second and sets the running speed. When it decides to jump it also picks the jump "
            "height and the spin, and from then on physics decides whether the dog catches it and lands.")
    hint = "A chime is a catch, a thud is a miss or a bad landing"
    stat_label = "Catch"
    vars = [
        ("caught", "1 if the dog catches the frisbee, else 0"),
        ("miss", "closest the mouth got to the frisbee, m (0 if caught)"),
        ("catch_height", "height of the catch, m"),
        ("spins", "full spins during the jump"),
        ("landing", "1 if it lands on its feet, else 0"),
        ("jump", "jump height, m"),
        ("run", "distance run before the jump, m"),
        ("jumped", "1 if the dog jumped at all, else 0"),
    ]
    presets = [
        ("Chase it", "10*caught - miss"),
        ("Only catches count", "caught"),
        ("High catch", "10*caught - miss + 4*catch_height"),
        ("Trick catch", "caught*(10 + 6*spins*landing) - miss"),
        ("Lazy dog", "-run"),
    ]
    obs_names = ["dog x", "dog y", "dog vx", "dog vy", "sin tilt", "cos tilt", "spin rate", "has jumped",
                 "time", "frisbee dx", "frisbee dy", "frisbee vx", "frisbee vy", "frisbee landed"]
    act_names = ["run speed", "jump now", "jump power", "spin"]
    settings = [
        ("throw", "Throw strength", "How hard the frisbee is thrown, × normal", 0.7, 1.3, 0.05, 1.0),
        ("float", "Frisbee float", "Higher floats longer before it drops, × normal", 0.6, 1.6, 0.05, 1.0),
        ("run_speed", "Dog top speed", "Fastest the dog can run, m/s", 3.0, 10.0, 0.1, 7.0),
        ("jump", "Jump power", "How high the dog can spring, × normal", 0.6, 1.5, 0.05, 1.0),
    ]
    setting_presets = [
        ("Long throws", {"throw": 1.3, "float": 1.4}),
        ("Floaty frisbee", {"float": 1.6}),
        ("Puppy", {"run_speed": 4, "jump": 0.7}),
        ("Super dog", {"run_speed": 10, "jump": 1.5}),
    ]

    def __init__(self):
        super().__init__()
        s = self.space = pymunk.Space()
        s.gravity = (0, -9.81)
        s.iterations = 12
        ground = pymunk.Segment(s.static_body, (-50, 0), (200, 0), 0.0)
        ground.friction = 1.0
        ground.elasticity = 0.05
        s.add(ground)
        self.body = pymunk.Body(20, pymunk.moment_for_box(20, (2 * HX, 2 * HY)))
        self.box = pymunk.Poly.create_box(self.body, (2 * HX, 2 * HY))
        self.box.elasticity = 1.0
        s.add(self.body, self.box)

    def reset(self, rng):
        cf = self.cfg
        self.fvx = float(rng.uniform(3.4, 5.0)) * cf["throw"]
        self.fvy = float(rng.uniform(2.8, 4.4)) * cf["throw"]
        self.fa = float(rng.uniform(2.2, 2.8)) / cf["float"]
        c = 1.6 - 0.03
        self.tf = (self.fvy + math.sqrt(self.fvy ** 2 + 4 * self.fa * c)) / (2 * self.fa)
        self.max_t = max(MAX_T, self.tf + 2.5)
        b = self.body
        b.position = (X0, HY)
        b.velocity = (0, 0)
        b.angle = 0
        b.angular_velocity = 0
        self.box.friction = 0.0
        self.k = 0
        self.t = 0.0
        self.jumped = False
        self.tj = -1.0
        self.w = -1.0
        self.th_jump = 0.0
        self.run_d = 0.0
        self.miss = 1e9
        self.tc = -1.0
        self.ch = 0.0
        self.tl = -1.0
        self.xl = 0.0
        self.spins = 0.0
        self.peak = 0.0
        self.still = 0.0
        self.speed = 0.0
        self.frames = []
        return self._obs()

    def fx(self, t):
        return -0.6 + self.fvx * t

    def fy(self, t):
        return max(0.03, 1.6 + self.fvy * t - self.fa * t * t)

    def _mouth(self):
        b = self.body
        c, s = math.cos(b.angle), math.sin(b.angle)
        return b.position.x + MOUTH[0] * c - MOUTH[1] * s, b.position.y + MOUTH[0] * s + MOUTH[1] * c

    def _obs(self):
        b = self.body
        mx, my = self._mouth()
        if self.tc >= 0:
            ft = self.tc
        else:
            ft = min(self.t, self.tf)
        landed = self.t >= self.tf
        fvx = 0.0 if landed else self.fvx
        fvy = 0.0 if landed else self.fvy - 2 * self.fa * ft
        return np.array([
            b.position.x / 5, b.position.y, b.velocity.x / 5, b.velocity.y / 5,
            math.sin(b.angle), math.cos(b.angle), b.angular_velocity / 10, float(self.jumped), self.t / 2,
            (self.fx(ft) - mx) / 5, self.fy(ft) - my, fvx / 5, fvy / 5, float(landed),
        ], dtype=np.float32)

    def step(self, a):
        b = self.body
        if not self.jumped:
            self.speed = self.cfg["run_speed"] * u01(a[0])
            if self.w < 0 and self.speed > 0.3:
                self.w = self.t
            if a[1] > 0 and self.t > 0.05:
                self.jumped = True
                self.tj = self.t
                self.run_d = b.position.x - X0
                self.th_jump = b.angle
                b.velocity = (self.speed, (2 + 5 * u01(a[2])) * self.cfg["jump"])
                b.angular_velocity = -14 * u01(a[3])
        for _ in range(SUBSTEPS):
            if self._substep():
                return self._obs(), True
        if self.jumped or self.t > self.tf + 0.3:
            # nothing left to decide: let physics finish the episode
            while not self._substep():
                pass
            return self._obs(), True
        return self._obs(), False

    def _substep(self):
        """One physics step. Returns True when the episode is over."""
        b = self.body
        t = self.t
        if self.k % REC_EVERY == 0:
            self.frames.extend((b.position.x, b.position.y, b.angle))
        if not self.jumped:
            b.velocity = (self.speed, min(0.0, b.velocity.y))
            b.angular_velocity = 0
        elif self.box.friction == 0 and b.position.y > HY + 0.06:
            self.box.friction = 0.3
        self.space.step(DT)
        self.k += 1
        self.t = t = self.k * DT
        th = b.angle
        if self.jumped and self.tl < 0 and t > self.tj + 0.08:
            low = b.position.y - HX * abs(math.sin(th)) - HY * abs(math.cos(th))
            if low < 0.03:
                self.tl, self.xl = t, b.position.x
                self.spins = -(th - self.th_jump) / TAU
        self.peak = max(self.peak, b.position.y - HY)
        if self.tc < 0 and t <= self.tf:
            mx, my = self._mouth()
            d = math.hypot(mx - self.fx(t), my - self.fy(t))
            self.miss = min(self.miss, d)
            if d < 0.3:
                self.tc, self.ch = t, self.fy(t)
        end = self.tc if self.tc >= 0 else self.tf
        if self.jumped and self.tl >= 0:
            slow = math.hypot(*b.velocity) < 0.08 and abs(b.angular_velocity) < 0.1
            self.still = self.still + DT if slow else 0.0
            if self.still > 0.4 and t > end + 0.6:
                return True
        if not self.jumped and t > self.tf + 0.3:
            return True
        return t >= self.max_t

    def metrics(self):
        caught = 1.0 if self.tc >= 0 else 0.0
        return {
            "caught": caught, "miss": 0.0 if caught else self.miss, "catch_height": self.ch,
            "spins": self.spins, "landing": 1.0 if math.cos(self.body.angle) > 0.92 else 0.0,
            "jump": self.peak, "run": self.run_d if self.jumped else self.body.position.x - X0,
            "jumped": float(self.jumped),
        }

    def replay(self):
        tl = self.tl if self.tl >= 0 else self.t
        xl = self.xl if self.tl >= 0 else self.body.position.x
        caught = self.tc >= 0
        E = self.tc if caught else self.tf
        tj = self.tj if self.jumped else self.t + 1
        return {"fr": rounded(self.frames), "fvx": round(self.fvx, 4), "fvy": round(self.fvy, 4), "fa": round(self.fa, 4),
                "tf": round(self.tf, 4), "tc": round(self.tc, 4), "tj": round(tj, 3), "w": round(max(self.w, 0), 3),
                "tl": round(tl, 3), "xl": round(xl, 3), "E": round(E, 3), "T": round(max(self.t, E + 1) + 0.4, 3)}
