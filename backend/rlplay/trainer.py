"""Rollouts and the three learning algorithms (PPO, REINFORCE, evolution strategies)."""
import math
import time

import numpy as np
import torch
from torch import nn

from .envs import GAMES
from .formula import compile_formula
from .nets import MLP, Policy, clean_spec, param_count, snapshot

ALGOS = ("ppo", "reinforce", "es")

DEFAULT_HP = {
    "episodes": 32,        # episodes per iteration (population size for ES)
    "lr": 3e-4,
    "gamma": 0.99,
    "lam": 0.95,           # PPO: GAE lambda
    "epochs": 8,           # PPO: passes over each batch
    "minibatch": 128,      # PPO
    "clip": 0.2,           # PPO
    "target_kl": 0.05,     # PPO: stop the epochs early past this KL (0 = off)
    "ent": 0.0,            # PPO / REINFORCE: entropy bonus
    "vf": 0.5,             # value-loss weight
    "max_grad": 0.5,
    "log_std": -0.5,       # starting exploration noise (log of the std)
    "baseline": True,      # REINFORCE: subtract the value net's estimate
    "sigma": 0.05,         # ES: weight noise
    "norm_reward": True,   # scale rewards by their running mean and std
}
ALGO_LR = {"ppo": 3e-4, "reinforce": 1e-3, "es": 0.03}
HP_LIMITS = {"episodes": (2, 512), "epochs": (1, 50), "minibatch": (8, 8192)}


def clean_hp(hp, algo):
    out = dict(DEFAULT_HP)
    out["lr"] = ALGO_LR[algo]
    for k, v in (hp or {}).items():
        if k not in DEFAULT_HP:
            continue
        d = DEFAULT_HP[k]
        v = bool(v) if isinstance(d, bool) else type(d)(v)
        if k in HP_LIMITS:
            v = max(HP_LIMITS[k][0], min(HP_LIMITS[k][1], v))
        out[k] = v
    if algo == "es":
        out["episodes"] += out["episodes"] % 2  # antithetic pairs
    return out


class RunningStat:
    def __init__(self):
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def push(self, xs):
        for x in xs:
            self.n += 1
            d = x - self.mean
            self.mean += d / self.n
            self.m2 += d * (x - self.mean)

    @property
    def std(self):
        return math.sqrt(self.m2 / self.n) if self.n > 1 else 1.0


class Episode:
    __slots__ = ("obs", "act", "logp", "metrics", "reward", "replay")

    def __init__(self):
        self.obs, self.act, self.logp = [], [], []


class Trainer:
    def __init__(self, game, formula, net, algo, hp, seed=None, settings=None):
        if game not in GAMES:
            raise ValueError(f"Unknown game {game!r}.")
        if algo not in ALGOS:
            raise ValueError(f"Unknown algorithm {algo!r}.")
        self.game = game
        self.Env = GAMES[game]
        self.envs = []
        probe = self.Env()
        self.obs_dim, self.act_dim = probe.obs_dim, probe.act_dim
        self.net = {"policy": clean_spec(net.get("policy")),
                    "value": clean_spec(net.get("value") or net.get("policy")),
                    "value_same": bool(net.get("value_same", True))}
        if self.net["value_same"]:
            self.net["value"] = self.net["policy"]
        self.seed = int(seed if seed is not None else np.random.SeedSequence().entropy % 2**31)
        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        self.algo = algo
        self.hp = clean_hp(hp, algo)
        self.policy = Policy(self.obs_dim, self.act_dim, self.net["policy"], self.hp["log_std"])
        self.set_formula(formula)
        self.set_settings(settings)
        self._build_learner()
        self.iteration = 0
        self.episodes = 0
        self.steps = 0
        self.history = []
        self.last = []
        self.first = None
        self.last_stats = None

    # ----- configuration -------------------------------------------------
    def set_formula(self, src):
        self.reward_fn = compile_formula(src, [v[0] for v in self.Env.vars])
        self.formula = src
        self.rstat = RunningStat()

    def set_settings(self, settings):
        """Change the scene (board height, gravity, ...). The policy keeps what it has learned."""
        self.settings = self.Env.clean_settings({**getattr(self, "settings", {}), **(settings or {})})
        for env in self.envs:
            env.configure(self.settings)

    def set_algo(self, algo, hp):
        """Switch algorithm or settings. The policy keeps what it has learned."""
        if algo not in ALGOS:
            raise ValueError(f"Unknown algorithm {algo!r}.")
        old = (self.algo, self.hp["log_std"])
        self.hp = clean_hp(hp, algo)
        if algo != self.algo:
            self.algo = algo
            self._build_learner()
        else:
            for g in self.opt.param_groups:
                g["lr"] = self.hp["lr"]
        if old[1] != self.hp["log_std"]:
            with torch.no_grad():
                self.policy.log_std.fill_(self.hp["log_std"])

    def _build_learner(self):
        self.value = None
        if self.algo in ("ppo", "reinforce"):
            self.value = MLP(self.obs_dim, 1, self.net["value"], out_gain=1.0)
            params = list(self.policy.parameters()) + list(self.value.parameters())
            self.opt = torch.optim.Adam(params, lr=self.hp["lr"], eps=1e-5)
        else:
            self.opt = torch.optim.Adam(self.policy.mu.parameters(), lr=self.hp["lr"])

    def info(self):
        return {"game": self.game, "formula": self.formula, "net": self.net, "algo": self.algo, "hp": self.hp,
                "settings": self.settings, "obsDim": self.obs_dim, "actDim": self.act_dim, "seed": self.seed,
                "params": {"policy": param_count(self.policy.mu), "value": param_count(self.value) if self.value else 0},
                "iteration": self.iteration, "episodes": self.episodes, "history": self.history, "last": self.last_stats}

    # ----- rollouts --------------------------------------------------------
    def _reward(self, m):
        r = self.reward_fn(m)
        return r if math.isfinite(r) else math.nan

    def rollout(self, n, deterministic=False, mus=None, seeds=None, progress=None):
        """Play ``n`` episodes in lockstep. ``mus`` gives one network per episode (used by ES).
        ``progress(phase, done, total)`` is called as episodes finish."""
        while len(self.envs) < n:
            env = self.Env()
            env.configure(self.settings)
            self.envs.append(env)
        eps = [Episode() for _ in range(n)]
        obs = []
        for i in range(n):
            rng = np.random.default_rng(seeds[i]) if seeds is not None else self.rng
            obs.append(self.envs[i].reset(rng))
        active = list(range(n))
        finished = 0
        while active:
            o = torch.as_tensor(np.stack([obs[i] for i in active]))
            with torch.no_grad():
                if mus is None:
                    a, logp = self.policy.act(o, deterministic)
                else:
                    a = torch.cat([mus[i](o[k:k + 1]) for k, i in enumerate(active)])
                    logp = torch.zeros(len(active))
            a_np = a.numpy()
            still = []
            for k, i in enumerate(active):
                e = eps[i]
                e.obs.append(obs[i])
                e.act.append(a_np[k])
                e.logp.append(float(logp[k]))
                obs[i], done = self.envs[i].step(np.clip(a_np[k], -1, 1))
                if done:
                    env = self.envs[i]
                    e.metrics = env.metrics()
                    e.reward = self._reward(e.metrics)
                    e.replay = env.replay()
                    finished += 1
                    if progress:
                        progress("play", finished, n)
                else:
                    still.append(i)
            active = still
        self.steps += sum(len(e.act) for e in eps)
        self.episodes += n
        return eps

    def _fix_nan(self, eps):
        ok = [e.reward for e in eps if not math.isnan(e.reward)]
        floor = (min(ok) - 1) if ok else 0.0
        for e in eps:
            if math.isnan(e.reward):
                e.reward = floor

    def _norm(self, rewards):
        if not self.hp["norm_reward"]:
            return np.asarray(rewards, dtype=np.float64)
        self.rstat.push(rewards)
        return np.clip((np.asarray(rewards) - self.rstat.mean) / (self.rstat.std + 1e-8), -10, 10)

    # ----- one learning step ----------------------------------------------
    def iterate(self, progress=None):
        t0 = time.perf_counter()
        if self.algo == "es":
            eps, extra = self._es(progress)
        else:
            eps = self.rollout(self.hp["episodes"], progress=progress)
            self._fix_nan(eps)
            extra = self._ppo(eps, progress) if self.algo == "ppo" else self._reinforce(eps, progress)
        self.iteration += 1
        eps.sort(key=lambda e: -e.reward)
        self.last = eps
        if self.first is None:
            self.first = eps[0]
        rewards = [e.reward for e in eps]
        names = [v[0] for v in self.Env.vars]
        stats = {
            "iter": self.iteration, "episodes": self.episodes, "steps": self.steps,
            "best": rewards[0], "mean": float(np.mean(rewards)), "worst": rewards[-1],
            "bestM": eps[0].metrics, "meanM": {k: float(np.mean([e.metrics[k] for e in eps])) for k in names},
            "std": None if self.algo == "es" else float(self.policy.log_std.detach().exp().mean()),
            "secs": time.perf_counter() - t0, **extra,
        }
        self.last_stats = stats
        self.history.append({k: stats[k] for k in ("iter", "best", "mean", "std") if k in stats}
                            | {k: extra[k] for k in ("pl", "vl", "ent") if k in extra})
        return stats

    def _flat(self, eps, adv_fn):
        obs = torch.as_tensor(np.concatenate([np.stack(e.obs) for e in eps]))
        act = torch.as_tensor(np.concatenate([np.stack(e.act) for e in eps]))
        logp = torch.as_tensor(np.concatenate([e.logp for e in eps]), dtype=torch.float32)
        with torch.no_grad():
            val = self.value(obs).squeeze(-1).numpy() if self.value is not None else np.zeros(len(obs))
        rn = self._norm([e.reward for e in eps])
        adv, ret = adv_fn(eps, rn, val)
        return obs, act, logp, torch.as_tensor(adv, dtype=torch.float32), torch.as_tensor(ret, dtype=torch.float32)

    def _gae(self, eps, rn, val):
        g, lam = self.hp["gamma"], self.hp["lam"]
        adv = np.zeros(len(val))
        i = 0
        for e, R in zip(eps, rn):
            n = len(e.act)
            v = val[i:i + n]
            last = 0.0
            for t in reversed(range(n)):
                r = R if t == n - 1 else 0.0
                nv = v[t + 1] if t + 1 < n else 0.0
                last = r + g * nv - v[t] + g * lam * last
                adv[i + t] = last
            i += n
        return adv, adv + val

    def _mc(self, eps, rn, val):
        g = self.hp["gamma"]
        ret = np.concatenate([R * g ** np.arange(len(e.act) - 1, -1, -1) for e, R in zip(eps, rn)])
        adv = ret - val if self.hp["baseline"] else ret
        return adv, ret

    def _step(self, loss):
        self.opt.zero_grad()
        loss.backward()
        params = [p for grp in self.opt.param_groups for p in grp["params"]]
        nn.utils.clip_grad_norm_(params, self.hp["max_grad"])
        self.opt.step()

    def _ppo(self, eps, progress=None):
        hp = self.hp
        obs, act, old_logp, adv, ret = self._flat(eps, self._gae)
        n = len(obs)
        pl = vl = ent = kl = cf = 0.0
        updates = 0
        stop = False
        for epoch in range(hp["epochs"]):
            if progress:
                progress("learn", epoch, hp["epochs"])
            perm = torch.randperm(n)
            for s in range(0, n, hp["minibatch"]):
                idx = perm[s:s + hp["minibatch"]]
                a = adv[idx]
                if len(idx) > 1:
                    a = (a - a.mean()) / (a.std() + 1e-8)
                d = self.policy.dist(obs[idx])
                logp = d.log_prob(act[idx]).sum(-1)
                ratio = (logp - old_logp[idx]).exp()
                p_loss = -torch.min(ratio * a, ratio.clamp(1 - hp["clip"], 1 + hp["clip"]) * a).mean()
                v_loss = ((self.value(obs[idx]).squeeze(-1) - ret[idx]) ** 2).mean()
                entropy = d.entropy().sum(-1).mean()
                self._step(p_loss + hp["vf"] * v_loss - hp["ent"] * entropy)
                with torch.no_grad():
                    lr_ = logp - old_logp[idx]
                    approx_kl = float(((lr_.exp() - 1) - lr_).mean())
                pl += p_loss.item(); vl += v_loss.item(); ent += entropy.item(); kl += approx_kl
                cf += float(((ratio - 1).abs() > hp["clip"]).float().mean())
                updates += 1
                if hp["target_kl"] > 0 and approx_kl > 1.5 * hp["target_kl"]:
                    stop = True
                    break
            if stop:
                break
        u = max(1, updates)
        return {"pl": pl / u, "vl": vl / u, "ent": ent / u, "kl": kl / u, "clipFrac": cf / u, "updates": updates}

    def _reinforce(self, eps, progress=None):
        hp = self.hp
        if progress:
            progress("learn", 0, 1)
        obs, act, _, adv, ret = self._flat(eps, self._mc)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        d = self.policy.dist(obs)
        logp = d.log_prob(act).sum(-1)
        entropy = d.entropy().sum(-1).mean()
        p_loss = -(logp * adv).mean()
        v_loss = ((self.value(obs).squeeze(-1) - ret) ** 2).mean() if hp["baseline"] else torch.zeros(())
        self._step(p_loss + hp["vf"] * v_loss - hp["ent"] * entropy)
        return {"pl": p_loss.item(), "vl": v_loss.item(), "ent": entropy.item()}

    def _es(self, progress=None):
        hp = self.hp
        mu = self.policy.mu
        theta = nn.utils.parameters_to_vector(mu.parameters()).detach()
        half = hp["episodes"] // 2
        noise = torch.randn(half, theta.numel())
        nets = []
        for k in range(half):
            for sign in (1, -1):
                m = MLP(self.obs_dim, self.act_dim, self.net["policy"])
                nn.utils.vector_to_parameters(theta + sign * hp["sigma"] * noise[k], m.parameters())
                nets.append(m)
        base = self.rng.integers(2**31, size=half)
        seeds = [int(s) for s in np.repeat(base, 2)]  # each +/- pair sees the same episode
        eps = self.rollout(len(nets), mus=nets, seeds=seeds, progress=progress)
        self._fix_nan(eps)
        r = np.array([e.reward for e in eps])
        ranks = np.empty(len(r))
        ranks[r.argsort()] = np.arange(len(r))
        shaped = torch.as_tensor(ranks / (len(r) - 1) - 0.5, dtype=torch.float32)
        diff = shaped[0::2] - shaped[1::2]
        grad = (diff[:, None] * noise).sum(0) / (len(r) * hp["sigma"])
        offset = 0
        for p in mu.parameters():
            k = p.numel()
            p.grad = (-grad[offset:offset + k] + 0.005 * theta[offset:offset + k]).view_as(p).clone()
            offset += k
        self.opt.step()
        return eps, {"gradNorm": float(grad.norm())}

    # ----- replays ---------------------------------------------------------
    def replay(self, mode="train", n=20):
        if mode == "eval":
            eps = self.rollout(n, deterministic=True)
            self._fix_nan(eps)
            eps.sort(key=lambda e: -e.reward)
        else:
            eps = self.last[:n]
        pack = lambda e: {"reward": e.reward, "m": e.metrics, **e.replay}
        return {"game": self.game, "mode": mode, "iter": self.iteration, "eps": [pack(e) for e in eps],
                "first": pack(self.first) if self.first is not None else None}

    def weights(self):
        return snapshot(self.policy)

    # ----- checkpoints -----------------------------------------------------
    def state(self):
        return {"game": self.game, "formula": self.formula, "net": self.net, "algo": self.algo, "hp": self.hp,
                "settings": self.settings, "seed": self.seed, "iteration": self.iteration, "episodes": self.episodes, "steps": self.steps,
                "history": self.history, "policy": self.policy.state_dict(),
                "value": self.value.state_dict() if self.value is not None else None,
                "rstat": (self.rstat.n, self.rstat.mean, self.rstat.m2)}

    @classmethod
    def from_state(cls, st):
        t = cls(st["game"], st["formula"], st["net"], st["algo"], st["hp"], st["seed"], st.get("settings"))
        t.policy.load_state_dict(st["policy"])
        if t.value is not None and st.get("value") is not None:
            t.value.load_state_dict(st["value"])
        t.iteration, t.episodes, t.steps, t.history = st["iteration"], st["episodes"], st["steps"], st["history"]
        t.rstat.n, t.rstat.mean, t.rstat.m2 = st["rstat"]
        return t
