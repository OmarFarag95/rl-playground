// The showcase film: plays recorded milestones of a training run on a fixed timeline.
// Open showcase.html to preview in real time; render.py drives it frame by frame with window.__step.
import { createStage } from '../frontend/stage.js';
import { createNet, lerpWeights } from './netfx.js';

const $ = id => document.getElementById(id);
const capture = location.search.includes('capture');
// a seeded Math.random, so splashes come out the same on every render
let seed = 11;
Math.random = () => { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; };

const data = await (await fetch('out/showcase.json')).json();
await document.fonts.ready;

const SUB = [
  'Random weights. It has never touched water.',
  'Five updates in. Still a beginner.',
  'Learning what a splash costs',
  'Straighter, cleaner, less water',
  'Nearly there',
  'A clean entry from a high board',
];
const PLAY_SPEED = 0.6;          // slow motion for the dives
const ease = p => p < 0.5 ? 2 * p * p : 1 - (-2 * p + 2) ** 2 / 2;
const fmt = (v, d = 0) => v.toFixed(d);
const css = getComputedStyle(document.documentElement);
const color = n => css.getPropertyValue('--' + n).trim();

let T = 0;
const stage = createStage({ canvas: $('stage'), flat: $('flat'), tags: [...document.querySelectorAll('.tag')], now: () => T, gl: { preserveDrawingBuffer: true } });
stage.setGame(data.game);
stage.readColors();
const net = createNet($('net'), { obs: data.obs, act: data.act, sizes: data.sizes, colors: Object.fromEntries(['accent', 'deep', 'ink', 'muted', 'line', 'panel', 'gold'].map(k => [k, color(k)])) });
$('formula').textContent = data.formula.replace('-', '−');
$('netNote').textContent = `${data.sizes.slice(1, -1).join(' × ')} tanh neurons · 20 decisions a second`;

const ch = data.chapters, N = ch.length;
$('dots').innerHTML = ch.map((c, i) => `<div class="dot" data-i="${i}"><i></i>Update ${c.iter}</div>`).join('');

// ----- the timeline -----
const segs = [];
let total = 0;
const add = (kind, dur, i) => { segs.push({ kind, start: total, dur, i }); total += dur; };
add('intro', 3.4, 0);
ch.forEach((c, i) => {
  add('card', 1.7, i);
  add('play', c.ep.T / PLAY_SPEED + 0.3, i);
  add('hold', 1.5, i);
  if (i < N - 1) add('learn', 3.6, i);
});
add('finale', 7, N - 1);

let cur = null, iterNow = 0;
function enter(s) {
  const c = ch[s.i];
  $('card').hidden = true; $('badge').hidden = true; $('caption').hidden = true;
  stage.setShowAll(false);
  if (s.kind === 'intro') {
    stage.load([c.ep]);
    showCard('Splash Lab', 'Watch a neural network learn to dive', `Nobody shows it how. It only gets one number after each dive: reward = −splash.`);
  } else if (s.kind === 'card') {
    stage.load([c.ep]);
    showCard(c.iter === 0 ? 'Before any learning' : c.iter === ch.at(-1).iter ? 'After training' : 'Checkpoint', `Update ${c.iter}`, SUB[s.i] || '');
  } else if (s.kind === 'learn') {
    stage.load(data.montages[s.i]); stage.setShowAll(true);
  } else if (s.kind === 'finale') {
    const a = ch[0].batch.splash, b = ch.at(-1).batch.splash;
    $('card').hidden = false;
    $('card').innerHTML = `<div class="k">${ch.at(-1).iter} updates · ${data.episodes.toLocaleString('en')} practice dives · ${data.trainSecs} seconds on a laptop CPU</div>
      <div class="h">From flop to clean entry</div>
      <div class="big"><div><b>${fmt(a)} L</b><span>average splash at first</span></div><div><b>→</b><span>&nbsp;</span></div><div><b>${fmt(b)} L</b><span>average splash now</span></div></div>
      <div class="s">Same game, same physics. Only the weights changed, nudged by reinforcement learning.</div>`;
  }
}
function showCard(k, h, s) {
  $('card').hidden = false;
  $('card').innerHTML = `<div class="k">${k}</div><div class="h">${h}</div><div class="s">${s}</div>`;
}

// network activations at the stage clock, blended between decisions
function netAt(ep, clock) {
  const ts = ep.times;
  let k = 0;
  while (k + 1 < ts.length && ts[k + 1] <= clock) k++;
  const next = Math.min(ts.length - 1, k + 1), span = ts[next] - ts[k] || 1, u = Math.max(0, Math.min(1, (clock - ts[k]) / span));
  const acts = ep.acts.map(layer => layer[k].map((v, j) => v + (layer[next][j] - v) * (next > k ? u : 0)));
  const after = clock - ts.at(-1) - 0.05;          // after the last decision the body goes limp
  return { acts, actions: ep.actions[k], flow: after > 0 ? Math.max(0, 1 - after * 1.5) : 1 };
}

function update(dt) {
  T += dt;
  let s = segs.find(x => T >= x.start && T < x.start + x.dur) || segs.at(-1);
  if (s !== cur) { cur = s; enter(s); }
  const p = Math.min(1, (T - s.start) / s.dur), c = ch[s.i];
  let weights = c.weights, net_ = { acts: null, actions: null, mode: 'forward', flow: 0.6 };
  // stage
  if (s.kind === 'play' || s.kind === 'hold') {
    stage.tick(dt * PLAY_SPEED);
    net_ = { ...netAt(c.ep, stage.clock), mode: 'forward' };
    if (s.kind === 'play') { $('badge').hidden = false; $('badge').className = 'badge'; $('badge').textContent = c.ep.label; }
    if (stage.clock >= c.ep.E) {
      const m = c.ep.m;
      $('caption').hidden = false;
      $('caption').textContent = `Splash ${fmt(m.splash)} L · ${m.headfirst ? 'head first' : 'feet first'}, ${fmt(m.entry_angle)}° off vertical · ${fmt(m.board, 1)} m board`;
    }
  } else if (s.kind === 'learn') {
    if (stage.tick(dt * 1.6)) stage.load(data.montages[s.i]);
    const q = ease(Math.min(1, p * 1.1));
    weights = lerpWeights(c.weights, ch[s.i + 1].weights, q);
    net_ = { acts: null, actions: null, mode: 'learn', flow: Math.sin(Math.PI * Math.min(1, p * 1.05)) };
    $('badge').hidden = false; $('badge').className = 'badge gold';
    $('badge').textContent = `▸▸ Practising · ${data.montages[s.i].length} of the tries along the way`;
  } else if (s.kind === 'intro' || s.kind === 'card') {
    stage.tick(0);
    net_ = { ...netAt(c.ep, 0), mode: 'forward', flow: 0.5 };
  } else if (s.kind === 'finale') {
    stage.tick(0);
    net_ = { ...netAt(c.ep, c.ep.times[Math.min(6, c.ep.times.length - 1)]), mode: 'forward', flow: 0.8 };
  }
  // where training stands
  iterNow = s.kind === 'learn' ? c.iter + (ch[s.i + 1].iter - c.iter) * ease(Math.min(1, p * 1.1)) : c.iter;
  const bSplash = s.kind === 'learn' ? c.batch.splash + (ch[s.i + 1].batch.splash - c.batch.splash) * ease(Math.min(1, p * 1.1)) : c.batch.splash;
  net.draw({ weights, ...net_, time: T, title: s.kind === 'learn' ? 'BACKPROPAGATION · EVERY WEIGHT NUDGED TOWARDS LESS SPLASH' : '' });
  drawCurve(iterNow);
  // header and chips
  const titles = { intro: ['Learning to dive', 'A neural network, a springboard and one rule'], card: [`Update ${c.iter}`, SUB[s.i]],
    play: [`Update ${c.iter}`, SUB[s.i]], hold: [`Update ${c.iter}`, SUB[s.i]],
    learn: [`Training · update ${Math.round(iterNow)}`, 'Practising: dive, score, adjust the weights, repeat'],
    finale: ['Mastered', 'Reinforcement learning, from scratch, on your own machine'] };
  $('title').textContent = titles[s.kind][0]; $('subtitle').textContent = titles[s.kind][1];
  $('cIter').textContent = Math.round(iterNow);
  $('cDives').textContent = (Math.round(iterNow) * 32).toLocaleString('en');
  $('cSplashL').textContent = s.i === N - 1 && s.kind !== 'learn' ? 'Avg splash, test' : 'Avg splash';
  $('cSplash').textContent = fmt(bSplash) + ' L';
  // milestone timeline
  const done = s.kind === 'learn' ? s.i + ease(Math.min(1, p * 1.1)) : s.i;
  $('tlFill').style.width = `calc((100% - 28px) * ${done / (N - 1)})`;
  document.querySelectorAll('#dots .dot').forEach((d, i) => d.classList.toggle('on', i <= Math.round(done) && (i <= s.i || s.kind === 'learn' && i <= done)));
}

// ----- learning curve -----
const cv = $('curve'), g = cv.getContext('2d');
const H_ = data.history;
const lo = Math.min(...H_.map(h => h.mean)), hi = Math.max(...H_.map(h => h.best));
function drawCurve(upTo) {
  const r = cv.getBoundingClientRect(); if (cv.width !== r.width) { cv.width = r.width; cv.height = r.height; }
  const w = r.width, h = r.height, L = 58, R = 16, Tp = 10, B = 26, n = H_.length;
  const px = i => L + i / n * (w - L - R), py = v => Tp + (hi - v) / (hi - lo) * (h - Tp - B);
  g.clearRect(0, 0, w, h);
  g.strokeStyle = color('line'); g.lineWidth = 1; g.beginPath(); g.moveTo(L, Tp); g.lineTo(w - R, Tp); g.moveTo(L, h - B); g.lineTo(w - R, h - B); g.stroke();
  g.font = '500 15px Barlow, sans-serif'; g.fillStyle = color('muted'); g.textAlign = 'right';
  g.fillText(fmt(hi), L - 8, Tp + 5); g.fillText(fmt(lo), L - 8, h - B + 5);
  let lastX = -99;
  ch.forEach(c => { const x = px(c.iter); g.fillStyle = color('line'); g.fillRect(x - 0.5, Tp, 1, h - Tp - B); if (x - lastX > 28) { g.fillStyle = color('muted'); g.textAlign = 'center'; g.fillText(c.iter, x, h - 6); lastX = x; } });
  const pts = H_.filter(q => q.iter <= upTo + 0.001);
  [['mean', color('muted'), 2], ['best', color('accent'), 2.5]].forEach(([k, c, lw]) => {
    g.strokeStyle = c; g.lineWidth = lw; g.lineJoin = 'round'; g.beginPath();
    pts.forEach((q, i) => i ? g.lineTo(px(q.iter), py(q[k])) : g.moveTo(px(q.iter), py(q[k]))); g.stroke();
    if (pts.length) { const q = pts.at(-1); g.fillStyle = c; g.beginPath(); g.arc(px(q.iter), py(q[k]), 5, 0, 7); g.fill(); }
  });
}

stage.resize();
window.__duration = total;
window.__step = dt => update(dt);
window.__ready = true;
if (!capture) {
  let last = performance.now();
  const loop = now => { update(Math.min(0.05, (now - last) / 1000)); last = now; if (T < total) requestAnimationFrame(loop); };
  requestAnimationFrame(loop);
} else update(0);
