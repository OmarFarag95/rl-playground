"""Common pieces for the games.

Every game is an episodic environment with continuous actions in [-1, 1].
The agent gets no reward while acting: when the episode ends the game reports
a dict of metrics, and the user's formula turns them into the one reward.
"""
import math

import numpy as np

DT = 1 / 120          # physics step
SUBSTEPS = 6          # physics steps per decision (20 decisions per second)
REC_EVERY = 2         # record a replay frame every 2 physics steps (60 fps)


class Env:
    key = ""
    name = ""
    lede = ""
    hint = ""
    stat_label = ""
    vars = []           # [(name, meaning)]
    presets = []        # [(label, formula)]
    obs_names = []
    act_names = []

    @property
    def obs_dim(self):
        return len(self.obs_names)

    @property
    def act_dim(self):
        return len(self.act_names)

    def reset(self, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def step(self, action: np.ndarray):
        """Apply one action (already clipped to [-1, 1]). Returns (obs, done)."""
        raise NotImplementedError

    def metrics(self) -> dict:
        raise NotImplementedError

    def replay(self) -> dict:
        """Everything the browser needs to redraw this episode."""
        raise NotImplementedError

    @classmethod
    def describe(cls):
        return {
            "key": cls.key, "name": cls.name, "lede": cls.lede, "hint": cls.hint,
            "statLabel": cls.stat_label, "vars": cls.vars, "presets": cls.presets,
            "obs": cls.obs_names, "act": cls.act_names,
        }


def u01(a):
    """Map an action in [-1, 1] to [0, 1]."""
    return (float(a) + 1.0) * 0.5


def rounded(xs, nd=3):
    return [round(float(x), nd) for x in xs]


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


TAU = 2 * math.pi
