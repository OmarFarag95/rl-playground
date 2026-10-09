"""FastAPI app: serves the browser UI and runs training sessions.

A session belongs to a browser tab, not to a socket: the tab keeps the session id
and sends it when it connects, so a reload picks up the same trainer, still
training if it was.
"""
import asyncio
import json
import math
import re
import time
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnected

from .envs import GAMES
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

    def _running_msg(self, reached=False):
        return {"t": "running", "on": self.running, "target": self.target, "reached": reached}

    async def handle(self, msg):
        kind = msg.get("t")
        if kind == "hello":
            resume = None
            if self.trainer is not None:
                resume = {**self.trainer.info(), "running": self.running, "target": self.target}
            await self.send({"t": "meta", "sid": self.sid,
                             "games": {k: g.describe() | {"flat": getattr(g, "flat", False)} for k, g in GAMES.items()},
                             "algos": ALGOS, "acts": list(ACTS), "inits": list(INITS), "hp": DEFAULT_HP,
                             "checkpoints": list_checkpoints(), "resume": resume})
            if self.trainer is not None:
                await self.send({"t": "weights", **self.trainer.weights()})
        elif kind == "setup":
            self.running = False
            try:
                trainer = await asyncio.to_thread(Trainer, msg["game"], msg["formula"], msg["net"], msg["algo"], msg.get("hp"), msg.get("seed"))
            except (ValueError, FormulaError, KeyError) as e:
                await self.send({"t": "error", "msg": str(e)})
                return
            async with self.lock:
                self.trainer = trainer
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
            async with self.lock:
                self.trainer = trainer
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
