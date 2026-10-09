"""Networks built from the layer list the user designs in the browser.

A spec looks like::

    {"layers": [{"units": 64, "act": "tanh", "norm": false}, ...], "init": "orthogonal"}
"""
import math

import torch
from torch import nn

# GELU uses its tanh form, so the NumPy copy in vec.py computes exactly the same thing
ACTS = {
    "relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU,
    "gelu": lambda: nn.GELU(approximate="tanh"), "silu": nn.SiLU, "softplus": nn.Softplus, "linear": nn.Identity,
}
INITS = ("orthogonal", "xavier", "kaiming", "default")
MAX_UNITS = 1024
MAX_LAYERS = 8


def clean_spec(spec):
    """Validate a user spec and fill defaults. Raises ValueError with a readable message."""
    spec = dict(spec or {})
    layers = spec.get("layers", [])
    if not isinstance(layers, list) or len(layers) > MAX_LAYERS:
        raise ValueError(f"Use at most {MAX_LAYERS} hidden layers.")
    out = []
    for i, l in enumerate(layers):
        units = int(l.get("units", 64))
        if not 1 <= units <= MAX_UNITS:
            raise ValueError(f"Layer {i + 1}: units must be between 1 and {MAX_UNITS}.")
        act = l.get("act", "tanh")
        if act not in ACTS:
            raise ValueError(f"Layer {i + 1}: unknown activation {act!r}.")
        out.append({"units": units, "act": act, "norm": bool(l.get("norm", False))})
    init = spec.get("init", "orthogonal")
    if init not in INITS:
        raise ValueError(f"Unknown initialisation {init!r}.")
    return {"layers": out, "init": init}


class MLP(nn.Module):
    def __init__(self, n_in, n_out, spec, out_gain=1.0):
        super().__init__()
        spec = clean_spec(spec)
        self.spec = spec
        mods, linears, prev = [], [], n_in
        for l in spec["layers"]:
            lin = nn.Linear(prev, l["units"])
            linears.append((lin, l["act"]))
            mods.append(lin)
            if l["norm"]:
                mods.append(nn.LayerNorm(l["units"]))
            mods.append(ACTS[l["act"]]())
            prev = l["units"]
        self.head = nn.Linear(prev, n_out)
        mods.append(self.head)
        self.net = nn.Sequential(*mods)
        for lin, act in linears:
            _init(lin, spec["init"], act, None)
        _init(self.head, spec["init"], "linear", out_gain)

    def forward(self, x):
        return self.net(x)

    def linears(self):
        return [m for m in self.net if isinstance(m, nn.Linear)]


def _init(lin, how, act, out_gain):
    """Initialise a layer. ``out_gain`` is set only for the output layer, to shrink its first outputs."""
    if how == "default":
        if out_gain is not None:
            lin.weight.data.mul_(out_gain)
        return
    gain = out_gain if out_gain is not None else {
        "relu": math.sqrt(2), "leaky_relu": math.sqrt(2), "gelu": math.sqrt(2), "silu": math.sqrt(2),
        "elu": math.sqrt(2), "tanh": 5 / 3}.get(act, 1.0)
    if how == "orthogonal":
        nn.init.orthogonal_(lin.weight, gain)
    elif how == "xavier":
        nn.init.xavier_uniform_(lin.weight, gain)
    elif how == "kaiming":
        nn.init.kaiming_normal_(lin.weight, nonlinearity="relu" if out_gain is None else "linear")
        if out_gain is not None:
            lin.weight.data.mul_(out_gain)
    nn.init.zeros_(lin.bias)


class Policy(nn.Module):
    """Gaussian policy: the network gives the mean, a learned vector gives the spread."""

    def __init__(self, obs_dim, act_dim, spec, log_std=-0.5):
        super().__init__()
        self.mu = MLP(obs_dim, act_dim, spec, out_gain=0.01)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std)))

    def dist(self, obs):
        mu = self.mu(obs)
        std = self.log_std.clamp(-5, 1).exp().expand_as(mu)
        return torch.distributions.Normal(mu, std)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        d = self.dist(obs)
        a = d.mean if deterministic else d.sample()
        return a, d.log_prob(a).sum(-1)


def param_count(module):
    return sum(p.numel() for p in module.parameters())


def snapshot(policy, max_nodes=20):
    """A small, display-sized copy of the policy weights for the network diagram."""
    layers = []
    for lin in policy.mu.linears():
        w = lin.weight.detach()
        rows = _pick(w.shape[0], max_nodes)
        cols = _pick(w.shape[1], max_nodes)
        layers.append({"shape": list(w.shape), "rows": rows, "cols": cols,
                       "w": [[round(float(w[r, c]), 3) for c in cols] for r in rows],
                       "b": [round(float(lin.bias.detach()[r]), 3) for r in rows]})
    return {"layers": layers, "std": [round(float(s), 3) for s in policy.log_std.detach().exp()]}


def _pick(n, k):
    if n <= k:
        return list(range(n))
    step = (n - 1) / (k - 1)
    return sorted({round(i * step) for i in range(k)})


def export(mlp, log_std=None):
    """A NumPy copy of an MLP (and the policy's spread), for worker processes (see vec.forward)."""
    layers = []
    for m in mlp.net:
        if isinstance(m, nn.Linear):
            layers.append(("linear", m.weight.detach().numpy().copy(), m.bias.detach().numpy().copy()))
        elif isinstance(m, nn.LayerNorm):
            layers.append(("norm", m.weight.detach().numpy().copy(), m.bias.detach().numpy().copy(), m.eps))
        else:
            name = next(k for k, v in ACTS.items() if isinstance(m, type(v())))
            layers.append(("act", name))
    return {"layers": layers, "log_std": None if log_std is None else log_std.detach().numpy().copy()}
