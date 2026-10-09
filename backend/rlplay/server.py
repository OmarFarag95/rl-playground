"""FastAPI app: serves the browser UI and runs training sessions.

A session belongs to a browser tab, not to a socket: the tab keeps the session id
and sends it when it connects, so a reload picks up the same trainer, still
training if it was.
"""
import asyncio
import copy
import json
import math
import re
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnected

from .envs import GAMES
from .envs.base import rounded
from .formula import FormulaError, compile_formula
from .nets import ACTS, INITS
from .trainer import ALGOS, DEFAULT_HP, Trainer

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
CHECKPOINTS = ROOT / "checkpoints"
WEIGHTS_EVERY = 0.5  # seconds between network-diagram updates
MAX_SESSIONS = 8     # oldest disconnected sessions are dropped beyond this
SID = re.compile(r"[A-Za-z0-9-]{8,64}")

app = FastAPI(title="Splash Lab")


def _json_safe(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


class Session:
    def __init__(self, sid):
        self.sid = sid
        self.ws: WebSocket | None = None
        self.detached_at = time.monotonic()
        self.trainer: Trainer | None = None
        self.lock = asyncio.Lock()
        self.running = False
        self.wake = asyncio.Event()
        self.last_weights = 0.0
        self.target = 0          # stop training at this many updates (0 = no limit)
        self.progress = None     # (phase, done, total) inside the current update
        self.live_task = None    # the network playing in real time, for live pokes
        self.live_env = None
        self.live_noise = False
        self.runner = asyncio.create_task(self.loop())

    async def send(self, msg):
        """Send to the attached tab, if any. While no tab is attached, training carries on unseen."""
        ws = self.ws
        if ws is None:
            return
        try:
            await ws.send_text(json.dumps(_json_safe(msg), separators=(",", ":")))
        except Exception:
            if self.ws is ws:
                self.detach(ws)

    def detach(self, ws):
        if self.ws is ws:
            self.ws = None
            self.detached_at = time.monotonic()

    async def call(self, fn, *args):
        """Run a trainer operation off the event loop, one at a time."""
        async with self.lock:
            return await asyncio.to_thread(fn, *args)

    async def loop(self):
        while True:
            if not (self.running and self.trainer):
                self.wake.clear()
                await self.wake.wait()
                continue
            await self.iterate()
            await asyncio.sleep(0)

    async def iterate(self):
        t = self.trainer

        def progress(phase, done, total):  # called from the training thread
            self.progress = (phase, done, total)

        self.progress = ("play", 0, t.hp["episodes"])
        ticker = asyncio.create_task(self._send_progress(t))
        try:
            stats = await self.call(t.iterate, progress)
        except Exception as e:  # keep the session alive and tell the user
            self.running = False
            await self.send({"t": "error", "msg": f"Training stopped: {e}"})
            await self.send(self._running_msg())
            return
        finally:
            ticker.cancel()
            self.progress = None
        if t is not self.trainer:
            return
        await self.send({"t": "stats", **stats})
        if self.running and self.target and t.iteration >= self.target:
            self.running = False
            await self.send(self._running_msg(reached=True))
        if time.monotonic() - self.last_weights > WEIGHTS_EVERY:
            self.last_weights = time.monotonic()
            await self.send({"t": "weights", **t.weights()})

    async def _send_progress(self, t):
        last = None
        while True:
            await asyncio.sleep(0.1)
            p = self.progress
            if p and p != last:
                last = p
                await self.send({"t": "progress", "iter": t.iteration + 1, "phase": p[0], "done": p[1], "total": p[2]})

    def stop_live(self):
        if self.live_task is not None:
            self.live_task.cancel()
        self.live_task = None
        self.live_env = None

    async def live(self, trainer):
        """Play the current network in real time and stream it, taking pokes as it goes.
        Each episode starts from a fresh copy of the network, so training can carry on meanwhile."""
        env = trainer.Env()
        env.configure(trainer.settings)
        rng = np.random.default_rng()
        loop = asyncio.get_running_loop()
        fr_len = getattr(env, "FR", 15)
        try:
            while True:
                policy = copy.deepcopy(trainer.policy)
                env.configure(trainer.settings)
                obs = env.reset(rng)
                self.live_env = env
                await self.send({"t": "live", "kind": "reset", "iter": trainer.iteration, "terrain": env.terrain(),
                                 "noise": self.live_noise})
                sent_f = sent_c = sent_e = sent_w = 0
                start = loop.time()
                k = 0
                done = False
                while not done:
                    with torch.no_grad():
                        a, _ = policy.act(torch.as_tensor(obs)[None], deterministic=not self.live_noise)
                    act = np.clip(a[0].numpy(), -1, 1)
                    obs, done = env.step(act)
                    k += 1
                    n = len(env.frames) // fr_len
                    m = env.metrics()
                    await self.send({"t": "live", "kind": "chunk", "fr": rounded(env.frames[sent_f * fr_len:]),
                                     "cfr": [rounded(f) for f in env.crate_frames[sent_c:]],
                                     "crates": [{"size": c["size"], "born": c["born"]} for c in env.crates],
                                     "events": env.events[sent_e:], "winds": env.winds[sent_w:],
                                     "m": m, "reward": trainer.reward_fn(m), "a": rounded(act, 2)})
                    sent_f, sent_c, sent_e, sent_w = n, len(env.crate_frames), len(env.events), len(env.winds)
                    await asyncio.sleep(max(0.0, start + n / 60 - loop.time()))  # real time: frames are 60 fps
                m = env.metrics()
                await self.send({"t": "live", "kind": "end", "m": m, "reward": trainer.reward_fn(m)})
                await asyncio.sleep(2.0)
        except asyncio.CancelledError:
            pass
        finally:
            if self.live_env is env:
                self.live_env = None

    def _running_msg(self, reached=False):
        return {"t": "running", "on": self.running, "target": self.target, "reached": reached}

    async def handle(self, msg):
        kind = msg.get("t")
        if kind == "hello":
            resume = None
            if self.trainer is not None:
                resume = {**self.trainer.info(), "running": self.running, "target": self.target}
            await self.send({"t": "meta", "sid": self.sid,
                             "games": {k: g.describe() | {"flat": getattr(g, "flat", False), "live": getattr(g, "live", False), "dense": getattr(g, "dense", False)} for k, g in GAMES.items()},
                             "algos": ALGOS, "acts": list(ACTS), "inits": list(INITS), "hp": DEFAULT_HP,
                             "checkpoints": list_checkpoints(), "resume": resume})
            if self.trainer is not None:
                await self.send({"t": "weights", **self.trainer.weights()})
        elif kind == "setup":
            self.running = False
            try:
                trainer = await asyncio.to_thread(Trainer, msg["game"], msg["formula"], msg["net"], msg["algo"], msg.get("hp"), msg.get("seed"), msg.get("settings"))
            except (ValueError, FormulaError, KeyError) as e:
                await self.send({"t": "error", "msg": str(e)})
                return
            self.stop_live()
            async with self.lock:
                old, self.trainer = self.trainer, trainer
            if old is not None:
                await asyncio.to_thread(old.close)
            await self.send({"t": "ready", **trainer.info()})
            await self.send({"t": "weights", **trainer.weights()})
            await self.send(self._running_msg())
        elif kind == "check":
            await self.send({"t": "check", "id": msg.get("id"), "msg": check_formula(msg["game"], msg["formula"])})
        elif self.trainer is None:
            await self.send({"t": "error", "msg": "Pick a game first."})
        elif kind == "formula":
            try:
                await self.call(self.trainer.set_formula, msg["formula"])
            except FormulaError as e:
                await self.send({"t": "formula", "ok": False, "msg": str(e)})
                return
            await self.send({"t": "formula", "ok": True, "formula": msg["formula"]})
        elif kind == "settings":
            await self.call(self.trainer.set_settings, msg.get("settings"))
            await self.send({"t": "settings", "game": self.trainer.game, "settings": self.trainer.settings})
        elif kind == "algo":
            try:
                await self.call(self.trainer.set_algo, msg["algo"], msg.get("hp"))
            except ValueError as e:
                await self.send({"t": "error", "msg": str(e)})
                return
            await self.send({"t": "algo", "algo": self.trainer.algo, "hp": self.trainer.hp, "params": self.trainer.info()["params"]})
        elif kind == "run":
            if "target" in msg:
                self.target = max(0, int(msg.get("target") or 0))
            on = bool(msg.get("on"))
            if on and self.target and self.trainer.iteration >= self.target:
                on = False
                await self.send({"t": "error", "msg": f"Already at {self.trainer.iteration} updates. Raise the goal to keep training."})
            self.running = on
            self.wake.set()
            await self.send(self._running_msg())
        elif kind == "step":
            if not self.running:
                await self.iterate()
        elif kind == "replay":
            mode = msg.get("mode", "train")
            if mode == "train" and not self.trainer.last:
                await self.send({"t": "replay", "mode": mode, "eps": [], "iter": 0})
                return
            r = await self.call(self.trainer.replay, mode, int(msg.get("n", 20)))
            await self.send({"t": "replay", **r})
        elif kind == "live":
            if msg.get("on") and getattr(self.trainer.Env, "live", False):
                self.live_noise = bool(msg.get("noise"))
                if self.live_task is None or self.live_task.done():
                    self.live_task = asyncio.create_task(self.live(self.trainer))
            else:
                self.stop_live()
                await self.send({"t": "live", "kind": "off"})
        elif kind == "poke":
            env = self.live_env
            if env is not None and hasattr(env, "poke"):
                x = msg.get("x")
                env.poke(str(msg.get("kind")), float(msg.get("value") or 0), None if x is None else float(x))
        elif kind == "save":
            name = safe_name(msg.get("name", ""))
            if not name:
                await self.send({"t": "error", "msg": "Give the checkpoint a name (letters, digits, - and _)."})
                return
            state = await self.call(self.trainer.state)
            CHECKPOINTS.mkdir(exist_ok=True)
            await asyncio.to_thread(torch.save, state, CHECKPOINTS / f"{name}.pt")
            await self.send({"t": "saved", "name": name, "checkpoints": list_checkpoints()})
        elif kind == "load":
            path = CHECKPOINTS / f"{safe_name(msg.get('name', ''))}.pt"
            if not path.is_file():
                await self.send({"t": "error", "msg": "That checkpoint no longer exists."})
                return
            self.running = False
            try:
                state = await asyncio.to_thread(torch.load, path, weights_only=False)
                trainer = await asyncio.to_thread(Trainer.from_state, state)
            except Exception as e:
                await self.send({"t": "error", "msg": f"Could not load the checkpoint: {e}"})
                return
            self.stop_live()
            async with self.lock:
                old, self.trainer = self.trainer, trainer
            if old is not None:
                await asyncio.to_thread(old.close)
            await self.send({"t": "ready", "loaded": path.stem, **trainer.info()})
            await self.send({"t": "weights", **trainer.weights()})
            await self.send(self._running_msg())


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_-]", "", str(s))[:60]


def list_checkpoints():
    if not CHECKPOINTS.is_dir():
        return []
    files = sorted(CHECKPOINTS.glob("*.pt"), key=lambda p: -p.stat().st_mtime)
    return [{"name": p.stem, "time": p.stat().st_mtime} for p in files]


def check_formula(game, src):
    try:
        compile_formula(src, [v[0] for v in GAMES[game].vars])
    except FormulaError as e:
        return str(e)
    return ""


SESSIONS: dict[str, Session] = {}


async def attach(ws, sid):
    """Connect a socket to its session, creating the session if the server has not seen it."""
    if not (isinstance(sid, str) and SID.fullmatch(sid)):
        sid = str(uuid.uuid4())
    s = SESSIONS.get(sid)
    if s is None:
        detached = sorted((x for x in SESSIONS.values() if x.ws is None), key=lambda x: x.detached_at)
        for old in detached[:max(0, len(SESSIONS) + 1 - MAX_SESSIONS)]:
            old.running = False
            old.runner.cancel()
            old.stop_live()
            if old.trainer is not None:
                old.trainer.close()
            del SESSIONS[old.sid]
        s = SESSIONS[sid] = Session(sid)
    elif s.ws is not None and s.ws is not ws:
        # the same session opened somewhere else (e.g. a duplicated tab): the newest one wins
        old = s.ws
        await s.send({"t": "taken"})
        s.detach(old)
        try:
            await old.close()
        except Exception:
            pass
    s.ws = ws
    return s


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    s = None
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if msg.get("t") == "hello":
                s = await attach(ws, msg.get("sid"))
            if s is not None:
                await s.handle(msg)
    except (WebSocketDisconnect, WebSocketDisconnected):
        # the tab went away, or we closed this socket because the same session reconnected
        pass
    finally:
        if s is not None:
            s.detach(ws)


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
