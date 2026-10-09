# Splash Lab

A reinforcement-learning playground. Four small physics games, a policy network you design, and a
reward formula you write. Training runs on your machine in PyTorch; the browser draws the episodes.

```
./run.sh --open          # http://127.0.0.1:8765
```

Options: `--port 8765`, `--host 127.0.0.1`, `--threads 4` (CPU threads for PyTorch; small networks are
fastest on a few threads).

## How to play

1. **Pick a game.** Each one is an episode the network plays from start to finish.
2. **Write a reward.** The formula is plain maths over the game's variables, such as `10*caught - miss`.
   The reward arrives once, at the end of each episode. You can change it while training: the network
   keeps what it has learned and adapts to the new goal.
3. **Design the network.** Add, remove and reorder layers, and set each layer's width, activation and LayerNorm.
   The critic (value network) can copy the policy's layout or have its own. *Build this network* starts again from random weights.
4. **Choose an algorithm.** PPO, REINFORCE or evolution strategies. Switching keeps the policy weights.
5. **Set a goal and train.** *Train for N updates* fills the progress bar and estimates the time left, and
   training pauses when it gets there (0 means no limit). The thin bar below it tracks the current update:
   first playing its episodes, then learning from them. Then watch. *Training episodes* shows the latest batch, exploration noise included.
   *Test the policy* runs fresh episodes with the noise turned off.
6. **Save a checkpoint** to `checkpoints/`. A checkpoint holds the weights, network, settings, formula and history.

## The games

| Game | The network sees | The network decides | Changes every episode |
|---|---|---|---|
| Diver | body pose, spin, height, speed | take-off power, drive and spin, then hips, knees and arms 20× a second | board height 1–3 m |
| Frisbee dog | the dog and the frisbee | running speed 20× a second, then when to jump, how high, how much spin | the throw |
| Pizza chef | the dough's flight, the hands | toss power, spin, wobble and drift, then where to move the hands | ceiling height, draught |
| Sandwich | where the bread and earlier pieces ended up | where to drop each of 5 pieces and how to turn it | bread position, how pieces slide |

The physics is 2D rigid-body simulation with [pymunk](https://www.pymunk.org/) at 120 Hz. The agent decides every 6 physics steps.

## Layout

```
backend/rlplay/
  envs/        the four games (Env: reset, step, metrics, replay)
  formula.py   safe parser for reward formulas (no eval)
  nets.py      MLP built from the layer spec, Gaussian policy, weight snapshots
  trainer.py   lock-step rollouts, PPO, REINFORCE, evolution strategies, checkpoints
  server.py    FastAPI: static files and one training session per WebSocket
frontend/
  app.js       UI, WebSocket client, charts, network designer
  stage.js     three.js scenes that replay episodes
  netview.js   network diagram coloured by live weights
```

Adding a game means subclassing `Env` in `backend/rlplay/envs/`, registering it in `envs/__init__.py`
and adding a scene in `frontend/stage.js` that draws its `replay()` data.

Training state lives on the server, in a session tied to your browser tab. Reloading the tab picks up
the same run, still training if it was. A new tab starts its own session. Restarting the server
ends every session, so save a checkpoint to keep a network.
