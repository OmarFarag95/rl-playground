"""Train the diver and record what the showcase video needs.

For a handful of milestones it saves one real episode, the policy weights at that
moment, and the activation of every neuron at every decision the network made,
so the video can show signals flowing through the network in step with the dive.

    python video/record.py            # writes video/out/showcase.json
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from rlplay.envs.base import DT, SUBSTEPS  # noqa: E402
from rlplay.envs.diver import WIND_UP  # noqa: E402
from rlplay.trainer import Trainer  # noqa: E402

GAME, FORMULA = "diver", "-splash"
NET = {"policy": {"layers": [{"units": 16, "act": "tanh"}, {"units": 16, "act": "tanh"}]}}
MILESTONES = [0, 5, 20, 60, 150, 300]
SEED = 3
OUT = ROOT / "video" / "out" / "showcase.json"


def weights(t):
    lins = t.policy.mu.linears()
    return {"W": [l.weight.detach().numpy().round(4).tolist() for l in lins],
            "b": [l.bias.detach().numpy().round(4).tolist() for l in lins],
            "std": t.policy.log_std.detach().exp().numpy().round(3).tolist()}


@torch.no_grad()
def activations(t, obs):
    """Every layer's output for each decision: inputs, hidden layers (after tanh), action means."""
    x = torch.as_tensor(np.stack(obs))
    layers = [x]
    for m in t.policy.mu.net:
        x = m(x)
        if not isinstance(m, torch.nn.Linear) or m is t.policy.mu.head:
            layers.append(x)
    return [l.numpy().round(3).tolist() for l in layers]


def decision_times(n):
    # decision 0 is the take-off (made before the springboard bounce); the rest come every 6 physics steps in the air
    return [0.0] + [round(WIND_UP + k * SUBSTEPS * DT, 4) for k in range(n - 1)]


def episode(t, e, label):
    return {"label": label, "reward": e.reward, "m": e.metrics, **e.replay,
            "acts": activations(t, e.obs), "times": decision_times(len(e.obs)),
            "actions": np.clip(np.stack(e.act), -1, 1).round(3).tolist()}


def pick(eps, q):
    """The episode at quantile ``q`` of reward (0 = worst)."""
    s = sorted(eps, key=lambda e: e.reward)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


def main():
    torch.set_num_threads(2)
    t = Trainer(GAME, FORMULA, NET, "ppo", {"episodes": 32}, seed=SEED)
    chapters, montages, t0 = [], [], time.time()
    final = MILESTONES[-1]
    # a few training episodes between milestones, for the fast-forward montages
    between = {m: set(np.linspace(a, m, 5, dtype=int)[1:-1].tolist()) for a, m in zip(MILESTONES, MILESTONES[1:])}
    mont = {m: [] for m in MILESTONES[1:]}
    rng = np.random.default_rng(SEED)
    for it in range(final + 1):
        w = weights(t) if it in MILESTONES else None
        if it < final:
            stats = t.iterate()
            batch = t.last
            for m, its in between.items():
                if it in its:
                    e = batch[rng.integers(len(batch))]
                    mont[m].append({"reward": e.reward, "m": e.metrics, **e.replay})
        if it in MILESTONES and it < final:
            q = {0: 0.0, 5: 0.25}.get(it, 0.5)  # update 0: the biggest flop of the batch; then typical tries
            e = pick(batch, q)
            chapters.append({"iter": it, "weights": w, "ep": episode(t, e, f"One of 32 tries at update {it}"),
                             "batch": {"mean": stats["mean"], "splash": stats["meanM"]["splash"],
                                       "angle": stats["meanM"]["entry_angle"]}})
            print(f"update {it:3d}: shown splash {e.metrics['splash']:.1f} L, batch mean {stats['meanM']['splash']:.1f} L", flush=True)
    secs = time.time() - t0
    # the finished network, tested without exploration noise, on a high board
    ev = [e for e in t.rollout(40, deterministic=True) if e.metrics["board"] > 2.3]
    e = pick(ev, 0.5)
    tests = t.rollout(64, deterministic=True)
    chapters.append({"iter": final, "weights": weights(t), "ep": episode(t, e, "Policy test, exploration noise off"),
                     "batch": {"mean": float(np.mean([x.reward for x in tests])),
                               "splash": float(np.mean([x.metrics["splash"] for x in tests])),
                               "angle": float(np.mean([x.metrics["entry_angle"] for x in tests]))}})
    print(f"update {final}: shown splash {e.metrics['splash']:.1f} L, test mean {chapters[-1]['batch']['splash']:.1f} L")
    env = t.Env
    data = {"game": GAME, "formula": FORMULA, "obs": env.obs_names, "act": env.act_names,
            "sizes": [env().obs_dim] + [l["units"] for l in NET["policy"]["layers"]] + [env().act_dim],
            "history": [{"iter": h["iter"], "best": h["best"], "mean": h["mean"]} for h in t.history],
            "chapters": chapters, "montages": [mont[m] for m in MILESTONES[1:]],
            "episodes": t.episodes, "trainSecs": round(secs)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB) after {secs:.0f} s of training")


if __name__ == "__main__":
    main()
