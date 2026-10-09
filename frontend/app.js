import { createStage, LOOKS, lookDefaults } from './stage.js';
import { drawNet } from './netview.js';

const $ = id => document.getElementById(id);
const store = {
  get(k, d) { try { const v = localStorage.getItem('splash.' + k); return v ? JSON.parse(v) : d; } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem('splash.' + k, JSON.stringify(v)); } catch (e) { /* private mode: ignore */ } },
};

const ALGO_INFO = {
  ppo: { name: 'PPO', hint: 'Proximal Policy Optimisation. Plays a batch of episodes, then makes several careful passes over them, keeping each change small. A sturdy default.' },
  reinforce: { name: 'REINFORCE', hint: 'The classic policy gradient. One step per batch: actions that led to above-average reward become more likely. Simple and noisy.' },
  es: { name: 'Evolution', hint: 'Evolution strategies. Tries many slightly different copies of the network without exploration noise and moves the weights towards the better ones. No gradients through the network.' },
};
// [key, label, help, step, algos]
const HP_FIELDS = [
  ['episodes', 'Episodes per update', 'More episodes give steadier but slower updates. For evolution this is the population.', 1, 'ppo reinforce es'],
  ['lr', 'Learning rate', 'How big each weight change is.', 'any', 'ppo reinforce es'],
  ['gamma', 'Discount γ', 'How much the reward at the end counts for earlier decisions.', 0.001, 'ppo reinforce'],
  ['lam', 'GAE λ', 'Trades bias for variance in the advantage estimate.', 0.01, 'ppo'],
  ['epochs', 'Epochs', 'Passes over each batch.', 1, 'ppo'],
  ['minibatch', 'Minibatch size', 'Decisions per gradient step.', 1, 'ppo'],
  ['clip', 'Clip range', 'How far the policy may move in one update.', 0.01, 'ppo'],
  ['target_kl', 'Target KL', 'Stop the epochs early if the policy changes more than this. 0 turns it off.', 0.001, 'ppo'],
  ['ent', 'Entropy bonus', 'Rewards keeping some randomness, to keep exploring.', 0.001, 'ppo reinforce'],
  ['vf', 'Value loss weight', 'How much the critic\'s error counts in the loss.', 0.05, 'ppo reinforce'],
  ['max_grad', 'Gradient clip', 'Largest allowed gradient norm.', 0.05, 'ppo reinforce'],
  ['log_std', 'Exploration (log σ)', 'Sets the exploration noise. Changing it resets the noise to this value.', 0.1, 'ppo reinforce'],
  ['baseline', 'Use the critic as a baseline', 'Subtract the value estimate to reduce noise.', null, 'reinforce'],
  ['norm_reward', 'Normalise rewards', 'Rescale rewards by their running mean and spread, so any formula scale works.', null, 'ppo reinforce'],
  ['sigma', 'Weight noise σ', 'How far each copy\'s weights are pushed.', 0.005, 'es'],
];
const NET_PRESETS = {
  'Tiny': [{ units: 16, act: 'tanh' }],
  'Small': [{ units: 64, act: 'tanh' }, { units: 64, act: 'tanh' }],
  'Wide ReLU': [{ units: 256, act: 'relu' }, { units: 256, act: 'relu' }],
  'Deep': [{ units: 32, act: 'elu' }, { units: 32, act: 'elu' }, { units: 32, act: 'elu' }, { units: 32, act: 'elu' }],
};

const S = {
  meta: null, game: store.get('game', 'diver'), algo: store.get('algo', 'ppo'), hp: store.get('hp', {}),
  net: store.get('net', { policy: { layers: NET_PRESETS.Small, init: 'orthogonal' }, value_same: true, value: { layers: NET_PRESETS.Small, init: 'orthogonal' } }),
  built: null, running: false, view: 'train', speed: 1, showAll: false, compare: false,
  goal: store.get('goal', 200), target: 0, times: [], upd: null,
  history: [], lastStats: null, snap: null, iter: 0, shownIter: -1, pending: false, idle: true, sig: 'std',
};
const stage = createStage({ canvas: $('stage'), flat: $('flat'), tags: [...document.querySelectorAll('.tag')], onNoGL: () => { $('nogl').hidden = false; $('stage').hidden = true; } });
let col = {};

// ---------- connection ----------
// The session id lives in sessionStorage: it survives reloads but each tab gets its own trainer.
const sid = (() => {
  const make = () => (crypto.randomUUID ? crypto.randomUUID() : Array.from({ length: 4 }, () => Math.random().toString(36).slice(2, 10)).join('-'));
  try { let v = sessionStorage.getItem('splash.sid'); if (!v) { v = make(); sessionStorage.setItem('splash.sid', v); } return v; }
  catch (e) { return make(); }
})();
let ws = null, retry = 0, taken = false;
function connect() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.onopen = () => { retry = 0; conn('Connected to the local trainer'); send({ t: 'hello', sid }); };
  ws.onclose = () => {
    S.running = false; showRunning();
    if (taken) return;
    conn('Lost the trainer. Is the server still running? Retrying…', true);
    setTimeout(connect, Math.min(5000, 500 * 2 ** retry++));
  };
  ws.onmessage = e => on(JSON.parse(e.data));
}
const send = m => { if (ws && ws.readyState === 1) ws.send(JSON.stringify(m)); };
function conn(text, bad) { $('conn').textContent = text; $('conn').classList.toggle('bad', !!bad); }
let toastT = 0;
function toast(text, bad) { const t = $('toast'); t.textContent = text; t.classList.toggle('bad', !!bad); t.hidden = false; clearTimeout(toastT); toastT = setTimeout(() => { t.hidden = true; }, 4200); }

function on(m) {
  switch (m.t) {
    case 'meta':
      S.meta = m; initUI(); listCheckpoints(m.checkpoints);
      if (m.resume) onReady({ ...m.resume, resumed: true });
      else setGame(S.meta.games[S.game] ? S.game : 'diver');
      break;
    case 'taken': taken = true; conn('This session was opened in another tab. Reload here to start a separate one.', true); break;
    case 'ready': onReady(m); break;
    case 'stats': onStats(m); break;
    case 'weights': S.snap = m; drawNetwork(); break;
    case 'replay': onReplay(m); break;
    case 'running':
      if (m.on && !S.running) S.times = []; // a pause would skew the time-per-update estimate
      S.running = m.on; if (m.target !== undefined) S.target = m.target; showRunning(); showProgress();
      if (m.reached) toast(`Reached the goal of ${m.target} updates. Training paused.`);
      break;
    case 'progress': onProgress(m); break;
    case 'settings': onSettings(m); break;
    case 'formula':
      if (m.ok) { $('err').textContent = ''; formulas()[S.game] = m.formula; store.set('formulas', formulas()); markPreset(); toast('New reward formula in use. The network keeps what it learned.'); }
      else $('err').textContent = m.msg;
      break;
    case 'check': if (m.id === checkId) $('err').textContent = m.msg; break;
    case 'algo': S.algo = m.algo; S.hp = m.hp; showAlgo(); setParams(m.params); drawNetwork(); drawCurves(); break;
    case 'saved': toast(`Saved checkpoint "${m.name}".`); listCheckpoints(m.checkpoints); break;
    case 'error': toast(m.msg, true); break;
  }
}

// ---------- game & setup ----------
const formulas = () => (S._f ||= store.get('formulas', {}));
const G = () => S.meta.games[S.game];
function setGame(key) { showGame(key); setup(); }
// show a game's text, presets and scene without touching the trainer
function showGame(key, formula) {
  S.game = key; store.set('game', key);
  const g = G();
  document.querySelectorAll('#games button').forEach(b => b.setAttribute('aria-pressed', b.dataset.g === key));
  $('lede').textContent = g.lede; $('sndHint').textContent = g.hint; $('sStatL').textContent = g.statLabel;
  $('all').hidden = !!g.flat;
  $('presets').innerHTML = '';
  g.presets.forEach(([name, f]) => {
    const b = document.createElement('button'); b.textContent = name; b.dataset.f = f;
    b.onclick = () => { $('formula').value = f; applyFormula(); };
    $('presets').appendChild(b);
  });
  $('formula').value = formula ?? (formulas()[key] || g.presets[0][1]);
  $('err').textContent = '';
  stage.setGame(key);
  stage.setLook(key, looks()[key] || {});
  renderWorld(); renderLook();
}
function setup() {
  stage.clear(); S.pending = false; S.shownIter = -1; S.snap = null;
  send({ t: 'setup', game: S.game, formula: $('formula').value, net: S.net, algo: S.algo, hp: S.hp, settings: worlds()[S.game] || {} });
}
function onReady(m) {
  S.snap = null;
  S.built = JSON.parse(JSON.stringify(m.net));
  if (m.loaded || m.resumed) {
    // the server's trainer wins over what this tab remembered
    S.net = JSON.parse(JSON.stringify(m.net)); store.set('net', S.net);
    stage.clear(); S.pending = false; S.shownIter = -1;
    showGame(m.game, m.formula); formulas()[m.game] = m.formula; store.set('formulas', formulas());
    if (m.loaded) toast(`Loaded "${m.loaded}" at update ${m.iteration}.`);
    else if (m.iteration) toast(`Picked up where you left off at update ${m.iteration}${m.running ? ', still training' : ''}.`);
  }
  S.hp = m.hp; S.algo = m.algo; store.set('hp', S.hp); store.set('algo', S.algo);
  if (m.settings) { worlds()[m.game] = m.settings; store.set('worlds', worlds()); renderWorld(); }
  S.history = m.history || []; S.iter = m.iteration; S.lastStats = m.last || null;
  $('sIter').textContent = m.iteration; $('sEps').textContent = fmtInt(m.episodes);
  $('sBest').textContent = S.lastStats ? fmt(S.lastStats.best) : '–';
  $('sStat').textContent = S.lastStats ? stage.stat(S.game, S.lastStats.bestM) : '–';
  setParams(m.params); renderLayers(); showAlgo(); markPreset(); vars(); drawCurves(); $('speedNote').textContent = '';
  if (m.running !== undefined) { S.running = m.running; showRunning(); }
  if (m.target !== undefined && m.resumed) { S.target = m.target; if (m.target) { S.goal = m.target; $('goal').value = m.target; } }
  S.times = []; S.upd = null; showProgress();
  requestReplay(m.iteration ? undefined : 'eval');
}
function setParams(p) {
  $('params').textContent = `${fmtInt(p.policy)} policy weights` + (p.value && S.algo !== 'es' ? ` · ${fmtInt(p.value)} critic weights` : '');
}

// ---------- training stats ----------
function onStats(m) {
  S.lastStats = m; S.iter = m.iter;
  S.times.push(performance.now()); if (S.times.length > 12) S.times.shift();
  S.upd = { iter: m.iter + 1, phase: 'play', done: 0, total: S.hp.episodes }; showProgress();
  S.history.push({ iter: m.iter, best: m.best, mean: m.mean, std: m.std, pl: m.pl, vl: m.vl, ent: m.ent });
  $('sIter').textContent = m.iter; $('sEps').textContent = fmtInt(m.episodes); $('sBest').textContent = fmt(m.best);
  $('sStat').textContent = stage.stat(S.game, m.bestM);
  const per = m.secs > 0 ? (S.hp.episodes / m.secs) : 0;
  $('speedNote').textContent = `${m.secs.toFixed(2)} s per update, about ${per.toFixed(0)} episodes a second` + (m.kl !== undefined ? ` · KL ${m.kl.toFixed(4)} · clipped ${(m.clipFrac * 100).toFixed(0)}%` : '');
  vars(); drawCurves();
  if (S.idle) requestReplay();
}
function vars() {
  const g = G(), s = S.lastStats;
  $('vars').innerHTML = g.vars.map(([n, d]) => `<tr><td>${n}</td><td>${d}</td><td class="n">${s ? fmt(s.bestM[n]) : '–'}</td><td class="n">${s ? fmt(s.meanM[n]) : '–'}</td></tr>`).join('');
}
const fmt = v => v === null || v === undefined ? '–' : Math.abs(v) >= 1e6 ? v.toExponential(1) : Number.isInteger(v) ? String(v) : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(Math.abs(v) < 1 ? 2 : 1);
const fmtInt = v => Number(v).toLocaleString('en');

// ---------- progress ----------
function onProgress(m) { S.upd = m; showProgress(); }
const dur = s => s < 60 ? `${Math.max(1, Math.round(s))} s` : s < 3600 ? `${Math.floor(s / 60)} min ${Math.round(s % 60)} s` : `${Math.floor(s / 3600)} h ${Math.round(s % 3600 / 60)} min`;
function setBar(el, frac) { const f = Math.max(0, Math.min(1, frac)); el.firstChild.style.width = (f * 100).toFixed(1) + '%'; el.setAttribute('aria-valuenow', Math.round(f * 100)); }
function showProgress() {
  const goal = S.running ? S.target : S.goal, gb = $('goalBar');
  // time per update, from the gaps between recent updates (falls back to the last reported duration)
  const t = S.times, per = t.length > 1 ? (t.at(-1) - t[0]) / (t.length - 1) / 1000 : S.lastStats ? S.lastStats.secs : 0;
  setBar(gb, goal ? S.iter / goal : 0);
  gb.classList.toggle('done', !!goal && S.iter >= goal);
  if (!goal) $('eta').textContent = `Update ${S.iter}` + (per && S.running ? ` · ${(1 / per).toFixed(per < 1 ? 1 : 2)} updates a second` : '');
  else if (S.iter >= goal) $('eta').textContent = `Done: ${S.iter} of ${goal} updates`;
  else $('eta').textContent = `Update ${S.iter} of ${goal}` + (per && S.running ? ` · about ${dur((goal - S.iter) * per)} left` : '');
  const u = S.upd;
  if (S.running && u) {
    setBar($('updBar'), u.total ? u.done / u.total : 0);
    $('updText').textContent = u.phase === 'play'
      ? `Update ${u.iter}: playing episodes, ${u.done} of ${u.total} done`
      : `Update ${u.iter}: learning from them` + (u.total > 1 ? `, pass ${u.done + 1} of ${u.total}` : '');
  } else {
    setBar($('updBar'), 0);
    $('updText').textContent = S.running ? 'Training…' : S.iter ? 'Paused. Press Train to continue.' : 'Press Train to start.';
  }
}

// ---------- replays ----------
function requestReplay(mode) {
  if (S.pending) return;
  mode = mode || S.view;
  // training episodes from before the first update, or from before a world change, have nothing new to show
  if (mode === 'train' && (S.iter === 0 || S.worldIter === S.iter)) mode = 'eval';
  if (mode === 'train' && S.shownIter === S.iter && !S.idle) return;
  S.pending = true; send({ t: 'replay', mode, n: 20 });
}
function onReplay(m) {
  S.pending = false;
  if (m.game && m.game !== S.game) return; // a late answer for the game we just left
  if (!m.eps || !m.eps.length) { S.idle = true; return; }
  S.idle = false; S.shownIter = m.mode === 'train' ? m.iter : -1;
  stage.load(m.eps, S.compare ? m.first : null);
  S.replayFirst = m.first;
  const what = m.mode === 'eval' ? `Policy test after ${m.iter} update${m.iter === 1 ? '' : 's'}, no exploration noise` : `Update ${m.iter}: best of ${S.hp.episodes} training episodes`;
  $('caption').textContent = `${what} · reward ${fmt(m.eps[0].reward)}`;
}
let last = 0;
function frame(now) {
  const dt = Math.min(0.05, (now - last) / 1000 || 0); last = now;
  const done = stage.tick(dt * S.speed);
  if (done && !S.pending) {
    // the replay is over: fetch something newer, or the next policy test
    if (S.view === 'eval') requestReplay('eval');
    else if (S.iter > S.shownIter) requestReplay('train');
    else S.idle = true;
  }
  requestAnimationFrame(frame);
}

// ---------- scene ----------
// World settings change the physics on the server; looks only change the picture and stay in this browser.
const worlds = () => (S._w ||= store.get('worlds', {}));
const looks = () => (S._l ||= store.get('looks', {}));
const worldDefaults = () => Object.fromEntries(G().settings.map(o => [o.key, o.default]));
const decimals = step => (String(step).split('.')[1] || '').length;
function sendWorld(settings) { worlds()[S.game] = settings; store.set('worlds', worlds()); renderWorld(); send({ t: 'settings', settings }); }
function renderWorld() {
  const g = G(), cur = { ...worldDefaults(), ...(worlds()[S.game] || {}) }, box = $('world');
  $('worldPresets').innerHTML = '';
  [['Standard', {}], ...(g.settingPresets || [])].forEach(([name, p]) => {
    const want = { ...worldDefaults(), ...p }, b = document.createElement('button');
    b.className = 'small'; b.textContent = name;
    b.setAttribute('aria-pressed', g.settings.every(o => Math.abs(want[o.key] - cur[o.key]) < 1e-9));
    b.onclick = () => sendWorld(want);
    $('worldPresets').appendChild(b);
  });
  box.innerHTML = '';
  g.settings.forEach(o => {
    const id = 'set_' + o.key, nd = decimals(o.step);
    const lab = document.createElement('label'); lab.htmlFor = id; lab.innerHTML = `${o.label}<small>${o.help}</small>`;
    const wrap = document.createElement('div'); wrap.className = 'slider';
    const inp = document.createElement('input'); Object.assign(inp, { type: 'range', id, min: o.lo, max: o.hi, step: o.step, value: cur[o.key] });
    const out = document.createElement('output'); out.htmlFor = id; out.textContent = (+cur[o.key]).toFixed(nd);
    inp.oninput = () => { out.textContent = (+inp.value).toFixed(nd); };
    inp.onchange = () => sendWorld({ ...cur, [o.key]: +inp.value });
    wrap.append(inp, out); box.append(lab, wrap);
  });
}
function onSettings(m) {
  if (m.game !== S.game) return;
  worlds()[S.game] = m.settings; store.set('worlds', worlds()); renderWorld();
  toast(S.running ? 'New world in use from the next update. The network keeps what it learned.' : 'New world set. Here is the current policy trying it.');
  // show the new world right away: until the next update, training episodes are still from the old one
  S.worldIter = S.iter;
  requestReplayNow();
}
function setLook(key, value) {
  const v = { ...(looks()[S.game] || {}), [key]: value };
  looks()[S.game] = v; store.set('looks', looks()); stage.setLook(S.game, v); renderLook();
}
function renderLook() {
  const cur = { ...lookDefaults(S.game), ...(looks()[S.game] || {}) }, box = $('look');
  box.innerHTML = '';
  LOOKS[S.game].forEach(o => {
    const id = 'look_' + o.key, lab = document.createElement('label');
    lab.htmlFor = id; lab.textContent = o.label;
    let ctl;
    if (o.type === 'check') {
      ctl = document.createElement('input'); Object.assign(ctl, { type: 'checkbox', id, checked: !!cur[o.key] });
      ctl.onchange = () => setLook(o.key, ctl.checked);
    } else if (o.type === 'select') {
      ctl = document.createElement('select'); ctl.id = id;
      ctl.innerHTML = o.options.map(([v, t]) => `<option value="${v}" ${v === cur[o.key] ? 'selected' : ''}>${t}</option>`).join('');
      ctl.onchange = () => setLook(o.key, ctl.value);
    } else {
      // swatches, plus a picker for any other colour; null is the theme colour
      ctl = document.createElement('div'); ctl.className = 'swatches'; ctl.setAttribute('role', 'group'); ctl.setAttribute('aria-label', o.label);
      const shown = c => c || col[o.token] || '#888888';
      o.options.forEach((c, k) => {
        const b = document.createElement('button'); b.style.background = shown(c);
        b.title = c ? c : 'Theme colour'; b.setAttribute('aria-label', `${o.label} ${k ? c : 'default'}`);
        b.setAttribute('aria-pressed', cur[o.key] === c);
        b.onclick = () => setLook(o.key, c);
        ctl.appendChild(b);
      });
      const pick = document.createElement('input'); pick.type = 'color'; pick.id = id; pick.title = 'Any colour';
      pick.value = /^#[0-9a-f]{6}$/i.test(shown(cur[o.key])) ? shown(cur[o.key]) : '#888888';
      pick.onchange = () => setLook(o.key, pick.value.toUpperCase());
      ctl.appendChild(pick);
      lab.htmlFor = '';
    }
    box.append(lab, ctl);
  });
}
$('sceneReset').onclick = () => {
  delete looks()[S.game]; store.set('looks', looks()); stage.setLook(S.game, {}); renderLook();
  sendWorld(worldDefaults());
};

// ---------- reward formula ----------
let checkId = 0, checkT = 0;
function applyFormula() { send({ t: 'formula', formula: $('formula').value }); }
function markPreset() { const v = $('formula').value.trim(); document.querySelectorAll('#presets button').forEach(b => b.setAttribute('aria-pressed', b.dataset.f === v)); }

// ---------- network designer ----------
function layerEditor(box, layers, onChange) {
  box.innerHTML = '';
  const io = (text) => { const d = document.createElement('div'); d.className = 'layer'; d.innerHTML = `<span class="ix"></span><span class="io">${text}</span>`; return d; };
  box.appendChild(io(`${G().obs.length} inputs`));
  layers.forEach((l, i) => {
    const row = document.createElement('div'); row.className = 'layer';
    row.innerHTML = `<span class="ix">${i + 1}</span>
      <input type="number" min="1" max="1024" value="${l.units}" aria-label="Units in layer ${i + 1}">
      <select aria-label="Activation of layer ${i + 1}">${S.meta.acts.map(a => `<option ${a === l.act ? 'selected' : ''}>${a}</option>`).join('')}</select>
      <label class="inline" title="Add a LayerNorm after this layer"><input type="checkbox" ${l.norm ? 'checked' : ''}> norm</label>
      <button class="small rm" title="Move up" ${i ? '' : 'disabled'} aria-label="Move layer ${i + 1} up">↑</button>
      <button class="small" title="Remove layer" aria-label="Remove layer ${i + 1}">×</button>`;
    const [units, act, norm] = row.querySelectorAll('input[type=number], select, input[type=checkbox]');
    const [up, rm] = row.querySelectorAll('button');
    units.onchange = () => { l.units = Math.max(1, Math.min(1024, Math.round(+units.value || 1))); units.value = l.units; onChange(); };
    act.onchange = () => { l.act = act.value; onChange(); };
    norm.onchange = () => { l.norm = norm.checked; onChange(); };
    up.onclick = () => { [layers[i - 1], layers[i]] = [layers[i], layers[i - 1]]; onChange(true); };
    rm.onclick = () => { layers.splice(i, 1); onChange(true); };
    box.appendChild(row);
  });
  box.appendChild(io(box.id === 'valLayers' ? '1 value' : `${G().act.length} actions`));
}
function renderLayers() {
  const changed = (redraw) => { store.set('net', S.net); if (redraw) renderLayers(); else markDirty(); drawNetwork(); };
  layerEditor($('polLayers'), S.net.policy.layers, changed);
  $('valSame').checked = S.net.value_same; $('valBox').hidden = S.net.value_same;
  if (!S.net.value) S.net.value = { layers: JSON.parse(JSON.stringify(S.net.policy.layers)), init: S.net.policy.init };
  layerEditor($('valLayers'), S.net.value.layers, changed);
  $('init').value = S.net.policy.init || 'orthogonal';
  markDirty(); drawNetwork();
}
function netKey(n) {
  const spec = s => s && { init: s.init || 'orthogonal', layers: s.layers.map(l => [l.units, l.act, !!l.norm]) };
  return JSON.stringify([spec(n.policy), !!n.value_same, n.value_same ? null : spec(n.value)]);
}
function markDirty() {
  const dirty = S.built && netKey(S.net) !== netKey(S.built);
  $('build').disabled = !dirty;
  $('buildHint').textContent = dirty ? 'You changed the layout. Building it starts learning from scratch.' : 'This is the network being trained.';
}
function drawNetwork() {
  if (!S.meta) return;
  const g = G();
  const built = S.built && netKey(S.net) === netKey(S.built);
  const snap = built && S.snap ? (S.algo === 'es' ? { ...S.snap, std: null } : S.snap) : null;
  drawNet($('net'), { obs: g.obs, act: g.act, layers: S.net.policy.layers, snap, colors: col });
}

// ---------- algorithm ----------
function showAlgo() {
  document.querySelectorAll('#algos button').forEach(b => b.setAttribute('aria-pressed', b.dataset.a === S.algo));
  $('algoHint').textContent = ALGO_INFO[S.algo].hint;
  const box = $('hp'); box.innerHTML = '';
  HP_FIELDS.filter(f => f[4].split(' ').includes(S.algo)).forEach(([k, label, help, step]) => {
    const id = 'hp_' + k, v = S.hp[k] ?? S.meta.hp[k];
    const lab = document.createElement('label'); lab.htmlFor = id; lab.innerHTML = `${label}<small>${help}</small>`;
    const inp = document.createElement('input'); inp.id = id;
    if (step === null) { inp.type = 'checkbox'; inp.checked = !!v; }
    else { inp.type = 'number'; inp.step = step; inp.value = v; }
    inp.onchange = () => { S.hp[k] = inp.type === 'checkbox' ? inp.checked : +inp.value; store.set('hp', S.hp); send({ t: 'algo', algo: S.algo, hp: S.hp }); };
    box.append(lab, inp);
  });
}

// ---------- charts ----------
function chart(canvas, series, opts) {
  const r = canvas.getBoundingClientRect(), d = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(r.width * d); canvas.height = Math.round(r.height * d);
  const g = canvas.getContext('2d'); g.setTransform(d, 0, 0, d, 0, 0);
  const w = r.width, h = r.height; g.clearRect(0, 0, w, h);
  g.font = '500 12px Barlow, sans-serif'; g.fillStyle = col.muted;
  const H = S.history.filter(p => series.some(s => Number.isFinite(p[s.key])));
  if (!H.length) { g.textAlign = 'center'; g.fillText(opts.empty, w / 2, h / 2 + 4); return; }
  // downsample long runs so the line stays readable
  const step = Math.max(1, Math.floor(H.length / Math.max(50, w / 2)));
  const pts = H.filter((_, i) => i % step === 0 || i === H.length - 1);
  let lo = Infinity, hi = -Infinity;
  pts.forEach(p => series.forEach(s => { const v = p[s.key]; if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }));
  if (hi - lo < 1e-9) { hi += 1; lo -= 1; }
  const L = 48, R = 10, T = 12, B = 20, n0 = pts[0].iter, n1 = pts.at(-1).iter;
  const px = it => L + (n1 === n0 ? 0 : (it - n0) / (n1 - n0)) * (w - L - R), py = v => T + (hi - v) / (hi - lo) * (h - T - B);
  g.strokeStyle = col.line; g.lineWidth = 1; g.beginPath(); g.moveTo(L, T); g.lineTo(w - R, T); g.moveTo(L, h - B); g.lineTo(w - R, h - B); g.stroke();
  g.textAlign = 'right'; g.fillText(fmt(hi), L - 6, T + 4); g.fillText(fmt(lo), L - 6, h - B + 4);
  g.textAlign = 'left'; g.fillText('Update ' + n0, L, h - 5); g.textAlign = 'right'; g.fillText('Update ' + n1, w - R, h - 5);
  series.forEach(s => {
    g.strokeStyle = s.color; g.lineWidth = s.width; g.lineJoin = 'round'; g.beginPath();
    let started = false;
    pts.forEach(p => { const v = p[s.key]; if (!Number.isFinite(v)) return; started ? g.lineTo(px(p.iter), py(v)) : g.moveTo(px(p.iter), py(v)); started = true; });
    g.stroke();
    const lastP = [...pts].reverse().find(p => Number.isFinite(p[s.key]));
    if (lastP) { g.fillStyle = s.color; g.beginPath(); g.arc(px(lastP.iter), py(lastP[s.key]), 3, 0, 7); g.fill(); }
  });
}
function drawCurves() {
  chart($('curve'), [{ key: 'mean', color: col.muted, width: 1.5 }, { key: 'best', color: col.accent, width: 2.5 }], { empty: 'The curve appears after the first update. Press Train.' });
  const none = { std: 'Evolution does not use exploration noise.', pl: 'Evolution has no policy loss.', vl: 'Only PPO and REINFORCE train a critic.', ent: 'Only PPO and REINFORCE report entropy.' };
  chart($('sig'), [{ key: S.sig, color: col.deep, width: 2 }], { empty: S.algo === 'es' ? none[S.sig] : 'Appears after the first update.' });
}

// ---------- checkpoints ----------
function listCheckpoints(list) {
  const sel = $('ckList');
  sel.innerHTML = list.length ? list.map(c => `<option value="${c.name}">${c.name} (${new Date(c.time * 1000).toLocaleString()})</option>`).join('') : '<option value="">No checkpoints yet</option>';
  $('load').disabled = !list.length;
}

// ---------- buttons ----------
function showRunning() { const b = $('train'); b.textContent = S.running ? 'Pause' : 'Train'; b.classList.toggle('on', S.running); $('step').disabled = S.running; }
function initUI() {
  $('games').innerHTML = '';
  Object.values(S.meta.games).forEach(g => { const b = document.createElement('button'); b.textContent = g.name; b.dataset.g = g.key; b.onclick = () => { if (g.key !== S.game) setGame(g.key); }; $('games').appendChild(b); });
  $('algos').innerHTML = '';
  S.meta.algos.forEach(a => { const b = document.createElement('button'); b.textContent = ALGO_INFO[a].name; b.dataset.a = a; b.onclick = () => { if (a === S.algo) return; S.algo = a; S.hp = {}; store.set('algo', a); store.set('hp', {}); send({ t: 'algo', algo: a, hp: {} }); drawCurves(); }; $('algos').appendChild(b); });
  $('init').innerHTML = S.meta.inits.map(i => `<option>${i}</option>`).join('');
  $('netPresets').innerHTML = '<span class="hint">Start from:</span>';
  Object.entries(NET_PRESETS).forEach(([name, layers]) => {
    const b = document.createElement('button'); b.className = 'small'; b.textContent = name;
    b.onclick = () => { S.net.policy.layers = JSON.parse(JSON.stringify(layers)); store.set('net', S.net); renderLayers(); };
    $('netPresets').appendChild(b);
  });
}
$('train').onclick = () => send({ t: 'run', on: !S.running, target: S.goal });
$('goal').value = S.goal;
$('goal').onchange = () => {
  S.goal = Math.max(0, Math.round(+$('goal').value || 0)); $('goal').value = S.goal; store.set('goal', S.goal);
  if (S.running) send({ t: 'run', on: true, target: S.goal }); else { S.target = S.goal; showProgress(); }
};
$('step').onclick = () => send({ t: 'step' });
$('reset').onclick = () => { if (S.iter === 0 || confirm('Start over with fresh random weights? The current network will be lost unless you save it.')) setup(); };
$('build').onclick = () => { S.net.policy.init = $('init').value; if (S.net.value) S.net.value.init = $('init').value; store.set('net', S.net); setup(); };
$('init').onchange = () => { S.net.policy.init = $('init').value; store.set('net', S.net); $('build').disabled = false; $('buildHint').textContent = 'Building it starts learning from scratch.'; };
$('polAdd').onclick = () => { const l = S.net.policy.layers; if (l.length >= 8) return toast('At most 8 hidden layers.', true); l.push({ ...(l.at(-1) || { units: 64, act: 'tanh' }) }); store.set('net', S.net); renderLayers(); };
$('valAdd').onclick = () => { const l = S.net.value.layers; if (l.length >= 8) return toast('At most 8 hidden layers.', true); l.push({ ...(l.at(-1) || { units: 64, act: 'tanh' }) }); store.set('net', S.net); renderLayers(); };
$('valSame').onchange = () => { S.net.value_same = $('valSame').checked; store.set('net', S.net); renderLayers(); };
$('apply').onclick = applyFormula;
$('formula').addEventListener('keydown', e => { if (e.key === 'Enter') applyFormula(); });
$('formula').addEventListener('input', () => { markPreset(); clearTimeout(checkT); checkT = setTimeout(() => send({ t: 'check', id: ++checkId, game: S.game, formula: $('formula').value }), 150); });
document.querySelectorAll('[data-speed]').forEach(b => b.onclick = () => { S.speed = +b.dataset.speed; document.querySelectorAll('[data-speed]').forEach(o => o.setAttribute('aria-pressed', o === b)); });
document.querySelectorAll('[data-view]').forEach(b => b.onclick = () => {
  S.view = b.dataset.view; document.querySelectorAll('[data-view]').forEach(o => o.setAttribute('aria-pressed', o === b));
  requestReplay();
});
document.querySelectorAll('[data-sig]').forEach(b => b.onclick = () => { S.sig = b.dataset.sig; document.querySelectorAll('[data-sig]').forEach(o => o.setAttribute('aria-pressed', o === b)); drawCurves(); });
$('cmp').onclick = () => { S.compare = !S.compare; $('cmp').setAttribute('aria-pressed', S.compare); $('firstKey').hidden = !S.compare; stage.setCompare(S.compare); if (S.compare && !S.replayFirst) toast('The first try appears once the network has done one update.'); requestReplayNow(); };
$('all').onclick = () => { S.showAll = !S.showAll; $('all').setAttribute('aria-pressed', S.showAll); $('ghostKey').hidden = !S.showAll; stage.setShowAll(S.showAll); };
$('snd').onclick = () => { const on = stage.sound.toggle(); if (on === null) { $('snd').textContent = 'No sound here'; return; } $('snd').setAttribute('aria-pressed', on); $('snd').textContent = on ? 'Sound on' : 'Sound off'; };
$('save').onclick = () => { const name = $('ckName').value.trim() || `${S.game}-${S.iter}`; send({ t: 'save', name }); };
$('load').onclick = () => { const n = $('ckList').value; if (n) send({ t: 'load', name: n }); };
function requestReplayNow() { S.shownIter = -1; S.idle = true; requestReplay(); }

function readColors() {
  const s = getComputedStyle(document.documentElement);
  ['accent', 'deep', 'ink', 'muted', 'line', 'panel', 'wall'].forEach(n => col[n] = s.getPropertyValue('--' + n).trim());
  stage.readColors(); drawCurves(); drawNetwork(); if (S.meta) renderLook();
}
let rT = 0;
window.addEventListener('resize', () => { clearTimeout(rT); rT = setTimeout(() => { stage.resize(); drawCurves(); drawNetwork(); }, 60); });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', readColors);
readColors();
showRunning();
connect();
requestAnimationFrame(frame);
