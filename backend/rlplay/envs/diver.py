import math

import numpy as np
import pymunk

from .base import DT, REC_EVERY, SUBSTEPS, TAU, Env, clamp, rounded, u01

# Body parts: mass, half height, half width, centre height above the feet.
# 0 torso+head, 1 thighs, 2 shins, 3 arms
SP = [(40, 0.425, 0.16, 1.325), (14, 0.22, 0.085, 0.68), (7, 0.23, 0.065, 0.23), (7, 0.31, 0.055, 1.19)]
M = sum(p[0] for p in SP)
CY = sum(p[0] * p[3] for p in SP) / M
# Joint motors: parent, child, stiffness, damping (hip, knee, shoulder)
JOINTS = [(0, 1, 700, 40), (1, 2, 250, 12), (0, 3, 200, 14)]
STRAIGHT = (0.0, 0.0, 2.8)   # hip, knee, shoulder targets with the body stretched, arms overhead
TUCKED = (2.1, -2.3, 1.0)    # knees to chest, hands on shins
LIMP = (0.5, -0.6, 1.6)
SPLASH_K = 10.5
TAIL = 0.6                   # seconds simulated after the torso enters the water
WIND_UP = 0.45               # springboard bounce before take-off (animation only)


class Diver(Env):
    key = "diver"
    name = "Diver"
    lede = ("A jointed diver leaves a springboard whose height changes every episode (1 to 3 m). "
            "At take-off the network picks the jump power, forward drive and spin. In the air it steers "
            "the hips, knees and arms 20 times a second, and a physics engine does the rest.")
    hint = 'A longer, louder "tishhh" means a worse entry'
    stat_label = "Best splash"
    vars = [
        ("splash", "water thrown out, litres"),
        ("entry_angle", "degrees off vertical at entry"),
        ("headfirst", "1 if head enters first, else 0"),
        ("flips", "full rotations in the air"),
        ("height", "peak height of the chest above water, m"),
        ("distance", "entry point from board tip, m"),
        ("airtime", "seconds in the air"),
        ("speed", "entry speed, m/s"),
        ("board", "springboard height this episode, m"),
    ]
    presets = [
        ("Clean entry", "-splash"),
        ("Head-first rip", "20*headfirst - splash"),
        ("Show dive", "10*flips + 15*headfirst - splash"),
        ("Long jump", "5*distance - splash"),
        ("Cannonball", "splash"),
    ]
    obs_names = ["take-off phase", "board height", "x", "y", "vx", "vy", "sin tilt", "cos tilt",
                 "spin rate", "hip angle", "knee angle", "arm angle", "time"]
    act_names = ["power / hip", "drive / knee", "spin / arms"]
    settings = [
        ("board_min", "Lowest board", "The springboard height is drawn between these two each episode, m", 0.5, 10.0, 0.1, 1.0),
        ("board_max", "Highest board", "Set both to the same value for a fixed board, m", 0.5, 10.0, 0.1, 3.0),
        ("spring", "Board springiness", "How hard the board throws the diver up, × normal", 0.5, 1.5, 0.05, 1.0),
        ("gravity", "Gravity", "9.81 on Earth, 3.71 on Mars, 24.8 on Jupiter, m/s²", 3.7, 25.0, 0.01, 9.81),
    ]
    setting_presets = [
        ("Olympic 10 m", {"board_min": 10, "board_max": 10}),
        ("Kiddie pool", {"board_min": 0.5, "board_max": 1}),
        ("Trampoline board", {"spring": 1.5}),
        ("Mars", {"gravity": 3.71}),
        ("Jupiter", {"gravity": 24.79}),
    ]

    def __init__(self):
        super().__init__()
        s = self.space = pymunk.Space()
        s.gravity = (0, -9.81)
        s.iterations = 12
        self.B = []
        for m, h, r, _ in SP:
            b = pymunk.Body(m, pymunk.moment_for_box(m, (2 * r, 2 * h)))
            self.B.append(b)
            s.add(b)
        for p, c, pa, pb in [(0, 1, -SP[0][1], SP[1][1]), (1, 2, -SP[1][1], SP[2][1]), (0, 3, SP[0][1] - 0.25, SP[3][1])]:
            j = pymunk.PivotJoint(self.B[p], self.B[c], (0, pa), (0, pb))
            j.collide_bodies = False
            s.add(j)

    def reset(self, rng):
        self.board = self.uniform(rng, "board_min", "board_max")
        self.g = self.cfg["gravity"]
        self.space.gravity = (0, -self.g)
        # give slow-motion worlds (low gravity, a springy board, a high board) time to land
        self.tmax = 6.0 * max(1.0, math.sqrt(9.81 / self.g)) * max(1.0, self.cfg["spring"]) * max(1.0, math.sqrt(self.board / 3))
        self.phase = 0
        self.t = 0.0
        self.k = 0
        self.entered = -1.0
        self.splash = 0.0
        self.peak = 0.0
        self.px = 0.0
        self.m = None
        self.targets = STRAIGHT
        self.frames = []
        self.done = False
        for i, b in enumerate(self.B):
            b.position = (0, self.board + SP[i][3])
            b.velocity = (0, 0)
            b.angle = 0
            b.angular_velocity = 0
        return self._obs()

    def _obs(self):
        B = self.B
        t0 = B[0]
        return np.array([
            1.0 if self.phase == 0 else 0.0, self.board / 3,
            t0.position.x / 5, t0.position.y / 5, t0.velocity.x / 5, t0.velocity.y / 5,
            math.sin(t0.angle), math.cos(t0.angle), t0.angular_velocity / 10,
            (B[1].angle - B[0].angle) / 3, (B[2].angle - B[1].angle) / 3, (B[3].angle - B[0].angle) / 3,
            self.t / 3,
        ], dtype=np.float32)

    def step(self, a):
        if self.phase == 0:
            g = [u01(x) for x in a]
            spring = self.cfg["spring"]
            self.amp = (0.1 + 0.2 * g[0]) * spring
            vy, vx, w = (1.5 + 5.5 * g[0]) * spring, 0.4 + 2.1 * g[1], -5.0 * g[2]
            for i, b in enumerate(self.B):
                ry = SP[i][3] - CY
                b.velocity = (vx - w * ry, vy)
                b.angular_velocity = w
            # springboard bounce frames, before physics takes over
            n = round(WIND_UP * 60)
            for k in range(n):
                yb = self.board - self.amp * math.sin(math.pi * k / 60 / WIND_UP)
                for i in range(4):
                    self.frames.extend((0.0, yb + SP[i][3], 0.0))
            self.phase = 1
            return self._obs(), False
        self.targets = tuple(s + (t - s) * u01(x) for s, t, x in zip(STRAIGHT, TUCKED, a))
        for _ in range(SUBSTEPS):
            self._substep()
            if self.entered >= 0:
                break
        if self.entered >= 0 or self.t >= self.tmax:
            while self.t < self.entered + TAIL and self.t < self.tmax + 0.5:
                self._substep()
            self._finish()
            return self._obs(), True
        return self._obs(), False

    def _substep(self):
        B = self.B
        if self.k % REC_EVERY == 0:
            for b in B:
                self.frames.extend((b.position.x, b.position.y, b.angle))
        limp = self.entered >= 0
        T, gain = (LIMP, 0.06) if limp else (self.targets, 1.0)
        torque = [0.0] * 4
        for j, (p, c, kp, kd) in enumerate(JOINTS):
            tq = gain * kp * (T[j] - (B[c].angle - B[p].angle)) - kd * (B[c].angular_velocity - B[p].angular_velocity)
            tq = clamp(tq, -600, 600)
            torque[c] += tq
            torque[p] -= tq
        prev_y = []
        for i, b in enumerate(B):
            prev_y.append(b.position.y)
            b.torque = torque[i]
            b.force = (0, SP[i][0] * self.g * 0.85) if b.position.y < 0 else (0, 0)
        self.space.step(DT)
        self.k += 1
        self.t = self.k * DT
        damp = 0.03 ** DT
        for i, b in enumerate(B):
            y = b.position.y
            if y < 0:
                b.velocity = b.velocity * damp
                b.angular_velocity *= damp
            if prev_y[i] > 0 >= y:
                self._enter(i, b)
        if B[0].position.y > self.peak:
            self.peak, self.px = B[0].position.y, B[0].position.x

    def _enter(self, i, b):
        m, h, r, _ = SP[i]
        vx, vy = b.velocity
        th = b.angle
        sp = math.hypot(vx, vy) or 1e-6
        ax, ay = -math.sin(th), math.cos(th)
        s = min(1.0, abs(ax * vy - ay * vx) / sp)
        self.splash += SPLASH_K * (sp * (4 * r * h * s + 0.25 * math.pi * r * r * math.sqrt(max(0.0, 1 - s * s)))
                                   + 2 * abs(b.angular_velocity) * h * h * r)
        if i == 0 and self.entered < 0:
            self.entered = self.t
            self.m = {
                "entry_angle": math.degrees(math.asin(min(1.0, abs(math.sin(th))))),
                "headfirst": 1.0 if math.cos(th) < 0 else 0.0,
                "flips": -th / TAU,
                "distance": b.position.x,
                "airtime": self.t,
                "speed": sp,
            }

    def _finish(self):
        if self.m is None:
            self.entered = self.t
            self.m = {"entry_angle": 90.0, "headfirst": 0.0, "flips": 0.0, "distance": 0.0, "airtime": self.t, "speed": 0.0}
        self.m.update(splash=self.splash, height=self.peak, board=self.board)
        self.done = True

    def metrics(self):
        return dict(self.m)

    def replay(self):
        E = self.entered + WIND_UP
        return {"fr": rounded(self.frames), "board": round(self.board, 3), "wt": WIND_UP, "amp": round(self.amp, 3),
                "E": round(E, 3), "T": round(E + 1.2, 3), "px": round(self.px, 3), "g": round(self.g, 2)}
