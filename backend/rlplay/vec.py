"""Run a batch of episodes either in this process or spread over worker processes.

LocalEnvs steps episodes in lockstep for Trainer.rollout:
    reset(seeds)        -> list of (observation, metrics or None), one per episode
    step(pairs)         -> for each (index, action): (obs, done, metrics or None, replay or None)
Metrics come back every step for games scored at every step (``dense``), otherwise only at the end.

PoolEnvs is for heavy games. Each worker gets a NumPy copy of the policy and plays whole episodes
on its own, so nothing waits on the main process between decisions:
    run(policies, seeds, deterministic, progress) -> one result dict per episode
"""
import math
import multiprocessing as mp
import os
from multiprocessing.connection import wait

import numpy as np


class LocalEnvs:
    def __init__(self, Env, settings):
        self.Env, self.settings, self.envs = Env, settings, []
        self.dense = getattr(Env, "dense", False)

    def configure(self, settings):
        self.settings = settings
        for e in self.envs:
            e.configure(settings)

    def reset(self, seeds):
        while len(self.envs) < len(seeds):
            e = self.Env()
            e.configure(self.settings)
            self.envs.append(e)
        out = []
        for i, s in enumerate(seeds):
            obs = self.envs[i].reset(np.random.default_rng(s))
            out.append((obs, self.envs[i].metrics() if self.dense else None))
        return out

    def step(self, pairs):
        out = []
        for i, a in pairs:
            env = self.envs[i]
            obs, done = env.step(a)
            m = env.metrics() if (done or self.dense) else None
            out.append((obs, done, m, env.replay() if done else None))
        return out

    def close(self):
        pass


# ----- a NumPy copy of a policy network (see nets.export) -------------------
def _gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


ACTS = {
    "relu": lambda x: np.maximum(x, 0), "tanh": np.tanh, "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "elu": lambda x: np.where(x > 0, x, np.expm1(np.minimum(x, 0))), "leaky_relu": lambda x: np.where(x > 0, x, 0.01 * x),
    "gelu": _gelu, "silu": lambda x: x / (1 + np.exp(-x)), "softplus": lambda x: np.logaddexp(0, x), "linear": lambda x: x,
}


def forward(spec, x):
    for layer in spec["layers"]:
        kind = layer[0]
        if kind == "linear":
            x = x @ layer[1].T + layer[2]
        elif kind == "norm":
            mu, var = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
            x = (x - mu) / np.sqrt(var + layer[3]) * layer[1] + layer[2]
        else:
            x = ACTS[layer[1]](x)
    return x


def play(env, spec, seed, deterministic, dense):
    """One whole episode with a NumPy policy. Gaussian sampling and log-probs match nets.Policy."""
    rng = np.random.default_rng([seed, 7])
    obs = env.reset(np.random.default_rng(seed))
    log_std = spec.get("log_std")
    std = np.exp(np.clip(log_std, -5, 1)) if log_std is not None else None
    O, A, L, M = [], [], [], [env.metrics()] if dense else None
    done = False
    while not done:
        mu = forward(spec, obs.astype(np.float32)[None])[0]
        if deterministic or std is None:
            a, lp = mu, 0.0
        else:
            a = mu + std * rng.standard_normal(mu.shape)
            lp = float(np.sum(-0.5 * ((a - mu) / std) ** 2 - np.log(std) - 0.5 * math.log(2 * math.pi)))
        O.append(obs)
        A.append(a.astype(np.float32))
        L.append(lp)
        obs, done = env.step(np.clip(a, -1, 1))
        if dense:
            M.append(env.metrics())
    return {"obs": np.stack(O), "act": np.stack(A), "logp": L, "metrics": env.metrics(), "steps": M, "replay": env.replay()}


def _worker(conn, game):
    from .envs import GAMES  # imported here: the worker starts fresh ("spawn")
    Env = GAMES[game]
    env = Env()
    dense = getattr(Env, "dense", False)
    while True:
        cmd, arg = conn.recv()
        if cmd == "configure":
            env.configure(arg)
        elif cmd == "run":
            out = []
            for i, seed, spec in arg["jobs"]:
                out.append((i, play(env, spec, seed, arg["deterministic"], dense)))
            conn.send(out)
        elif cmd == "close":
            conn.close()
            return


class PoolEnvs:
    """Episodes dealt round-robin to worker processes, each playing its share start to finish."""

    def __init__(self, Env, settings, workers):
        ctx = mp.get_context("spawn")
        self.dense = getattr(Env, "dense", False)
        self.conns, self.procs = [], []
        for _ in range(workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, Env.key), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        self.configure(settings)

    def configure(self, settings):
        for c in self.conns:
            c.send(("configure", settings))

    def run(self, policies, seeds, deterministic=False, progress=None):
        """``policies`` is one spec for every episode, or a list with one spec per episode."""
        n, w = len(seeds), len(self.conns)
        per = policies if isinstance(policies, list) else [policies] * n
        jobs = [[] for _ in range(w)]
        for i in range(n):
            jobs[i % w].append((i, seeds[i], per[i]))
        busy = {}
        for k, c in enumerate(self.conns):
            if jobs[k]:
                c.send(("run", {"jobs": jobs[k], "deterministic": deterministic}))
                busy[c] = k
        results, done = [None] * n, 0
        while busy:
            for c in wait(list(busy)):
                for i, r in c.recv():
                    results[i] = r
                    done += 1
                del busy[c]
                if progress:
                    progress("play", done, n)
        return results

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=1)
            if p.is_alive():
                p.terminate()


def make_envs(Env, settings):
    """Heavy games (``Env.parallel``) get worker processes; light ones are faster in-process."""
    if getattr(Env, "parallel", False):
        workers = min(10, max(1, (os.cpu_count() or 2) - 2))
        if workers > 1:
            return PoolEnvs(Env, settings, workers)
    return LocalEnvs(Env, settings)
