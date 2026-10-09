import math

import numpy as np

from .base import Env, rounded

PIECES = ["lettuce", "ham", "cheese", "tomato", "top slice"]
REACH = 0.8


class Sandwich(Env):
    key = "sandwich"
    name = "Sandwich"
    flat = True
    lede = ("The maker drops five pieces one at a time: lettuce, ham, cheese, tomato and the top slice. "
            "The bread sits in a slightly different spot every time and each piece slides a bit when it "
            "lands, so the network looks at where everything ended up before placing the next one.")
    hint = "A louder squelch means a messier sandwich"
    stat_label = "Covered"
    vars = [
        ("coverage", "% of the bread covered by fillings"),
        ("overhang", "% of the fillings hanging outside the bread"),
        ("lid_offset", "how far the top slice is off centre, cm"),
        ("lid_twist", "how far the top slice is turned, degrees"),
        ("tallest", "most fillings stacked on one spot"),
    ]
    presets = [
        ("Neat sandwich", "coverage - 2*overhang - 5*lid_offset - lid_twist/3"),
        ("Cover the bread", "coverage"),
        ("Nothing hangs out", "-overhang"),
        ("Just close it", "-5*lid_offset - lid_twist"),
        ("Tower", "10*tallest - overhang"),
    ]
    obs_names = ([f"next: {p}" for p in PIECES] + ["bread x", "bread y"]
                 + [f"{p} {c}" for p in PIECES[:4] for c in ("x", "y", "turn")])
    act_names = ["drop x", "drop y", "turn"]
    settings = [
        ("bread", "Bread wander", "How far from the centre the bread can sit, m", 0.0, 0.4, 0.01, 0.18),
        ("slide", "Slippery fillings", "How far a piece slides after it lands, m", 0.0, 0.25, 0.01, 0.05),
        ("twist", "Twisty fillings", "How much a piece turns by itself when it lands, degrees", 0.0, 30.0, 0.5, 3.5),
    ]
    setting_presets = [
        ("Steady hands", {"bread": 0, "slide": 0, "twist": 0}),
        ("Butter fingers", {"slide": 0.25, "twist": 30}),
        ("Wandering bread", {"bread": 0.4}),
    ]

    def reset(self, rng):
        self.rng = rng
        b = self.cfg["bread"]
        self.bx, self.by = (float(v) for v in rng.uniform(-b, b, 2))
        self.items = []
        return self._obs()

    def _obs(self):
        o = np.zeros(self.obs_dim, dtype=np.float32)
        j = len(self.items)
        if j < 5:
            o[j] = 1
        o[5], o[6] = self.bx / 0.18, self.by / 0.18
        for i, it in enumerate(self.items[:4]):
            o[7 + 3 * i: 10 + 3 * i] = (it["x"] / REACH, it["y"] / REACH, it["r"] / (math.pi / 2))
        return o

    def step(self, a):
        n = self.rng.normal(0, 1, 3)
        sl, tw = self.cfg["slide"], math.radians(self.cfg["twist"])
        self.items.append({"x": float(a[0]) * REACH + sl * n[0], "y": float(a[1]) * REACH + sl * n[1],
                           "r": float(a[2]) * math.pi / 2 + tw * n[2]})
        return self._obs(), len(self.items) == 5

    def metrics(self):
        it = [{"x": i["x"] - self.bx, "y": i["y"] - self.by, "r": i["r"]} for i in self.items]
        xs = -1.45 + 0.1 * np.arange(30)
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        cov = [np.hypot(X - it[0]["x"], Y - it[0]["y"]) < 0.42, np.hypot(X - it[1]["x"], Y - it[1]["y"]) < 0.36]
        c, s = math.cos(it[2]["r"]), math.sin(it[2]["r"])
        dx, dy = X - it[2]["x"], Y - it[2]["y"]
        cov.append((np.abs(dx * c + dy * s) < 0.3) & (np.abs(dy * c - dx * s) < 0.3))
        c, s = math.cos(it[3]["r"]) * 0.2, math.sin(it[3]["r"]) * 0.2
        cov.append((np.hypot(X - it[3]["x"] - c, Y - it[3]["y"] - s) < 0.19) | (np.hypot(X - it[3]["x"] + c, Y - it[3]["y"] + s) < 0.19))
        count = sum(m.astype(int) for m in cov)
        bread = (np.abs(X) < 0.5) & (np.abs(Y) < 0.55)
        filled = count > 0
        nb, nc, nf, no = bread.sum(), (bread & filled).sum(), filled.sum(), (filled & ~bread).sum()
        lid = it[4]
        return {"coverage": float(100 * nc / nb), "overhang": float(100 * no / nf) if nf else 0.0,
                "lid_offset": math.hypot(lid["x"], lid["y"]) * 12, "lid_twist": abs(math.degrees(lid["r"])),
                "tallest": float(count.max())}

    def replay(self):
        return {"it": [rounded((i["x"], i["y"], i["r"]), 4) for i in self.items], "bx": round(self.bx, 4),
                "by": round(self.by, 4), "E": 2.55, "T": 3.9}
