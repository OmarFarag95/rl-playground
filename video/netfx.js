// Animated policy network for the showcase video.
// forward: neurons light up with their activation and dots stream along each weight in
//          proportion to the signal it carries (weight × incoming activation).
// learn:   golden dots run backwards (backpropagation) while the weights morph to their new values.

const fract = x => x - Math.floor(x);
const hash = (a, b, c) => fract(Math.sin(a * 127.1 + b * 311.7 + c * 74.7) * 43758.5453);

export function createNet(canvas, { obs, act, sizes, colors, scale: k = 1 }) {
  const g = canvas.getContext('2d');
  let W = 0, H = 0;
  function fit() {
    const r = canvas.getBoundingClientRect(), d = 1;
    canvas.width = r.width * d; canvas.height = r.height * d; g.setTransform(d, 0, 0, d, 0, 0);
    W = r.width; H = r.height;
  }
  fit();
  const padL = 130 * k, padR = 170 * k, top = 40 * k, bot = 16 * k;
  const x = i => padL + i / (sizes.length - 1) * (W - padL - padR);
  const y = (n, j) => n === 1 ? (top + H - bot) / 2 : top + j / (n - 1) * (H - top - bot);
  const mix = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);
  const hex = h => { const v = parseInt(h.slice(1), 16); return [(v >> 16) & 255, (v >> 8) & 255, v & 255]; };
  const C = Object.fromEntries(Object.entries(colors).map(([k, v]) => [k, hex(v)]));
  const rgba = (c, a) => `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a})`;

  // weights: {W: [layer][out][in]}, acts: [layer][node] or null, actions: [act] or null
  function draw({ weights, acts, actions, mode, time, flow = 1, title }) {
    g.clearRect(0, 0, W, H);
    const L = weights.W.length;
    // edges
    for (let l = 0; l < L; l++) {
      const w = weights.W[l], a = sizes[l], b = sizes[l + 1], inp = acts ? acts[l] : null;
      let max = 1e-6;
      const sig = w.map(row => row.map((v, j) => { const s = inp ? v * Math.tanh(inp[j]) : v; max = Math.max(max, Math.abs(s)); return s; }));
      for (let i = 0; i < b; i++) for (let j = 0; j < a; j++) {
        const s = sig[i][j] / max, m = Math.abs(s), c = s >= 0 ? C.accent : C.deep;
        const x1 = x(l), y1 = y(a, j), x2 = x(l + 1), y2 = y(b, i);
        g.strokeStyle = rgba(c, mode === 'learn' ? 0.06 + 0.4 * m * m : 0.04 + 0.55 * m * m);
        g.lineWidth = 0.5 + 2.2 * m;
        g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke();
        // moving dots: forward carries the signal, learn carries the gradient back
        if (m > 0.22 && flow > 0) {
          const h = hash(l, i, j), sp = mode === 'learn' ? 1.6 : 0.9 + 0.8 * m;
          const f = mode === 'learn' ? 1 - fract(time * sp + h) : fract(time * sp + h);
          const px = x1 + (x2 - x1) * f, py = y1 + (y2 - y1) * f;
          g.fillStyle = mode === 'learn' ? rgba(C.gold, Math.min(1, 0.35 + m) * flow) : rgba(c, Math.min(1, 0.25 + m) * flow);
          g.beginPath(); g.arc(px, py, 1.6 + 2.6 * m, 0, 7); g.fill();
        }
      }
    }
    // neurons
    sizes.forEach((n, l) => {
      for (let j = 0; j < n; j++) {
        const v = acts ? Math.tanh(acts[l][j]) : 0, m = Math.abs(v), c = v >= 0 ? C.accent : C.deep;
        const cx = x(l), cy = y(n, j), r = (l === 0 || l === sizes.length - 1 ? 8 : 9) * k;
        if (mode !== 'learn' && m > 0.55) { g.shadowColor = rgba(c, 0.9); g.shadowBlur = 18 * m; }
        g.fillStyle = rgba(mix(C.panel, c, mode === 'learn' ? 0.15 * m : m), 1);
        g.strokeStyle = rgba(C.ink, 0.75); g.lineWidth = 1.2;
        g.beginPath(); g.arc(cx, cy, r, 0, 7); g.fill(); g.shadowBlur = 0; g.stroke();
      }
    });
    // column titles
    g.font = `600 ${15 * k}px "Barlow Condensed", sans-serif`; g.fillStyle = rgba(C.muted, 1); g.textAlign = 'center';
    sizes.forEach((n, l) => g.fillText((l === 0 ? 'SEES' : l === sizes.length - 1 ? 'DOES' : `HIDDEN ${l}`), x(l), 18 * k));
    // input labels
    g.font = `400 ${14 * k}px Barlow, sans-serif`; g.textAlign = 'right'; g.fillStyle = rgba(C.ink, 1);
    obs.forEach((name, j) => g.fillText(name, x(0) - 16 * k, y(sizes[0], j) + 5 * k));
    // outputs with a bar for the chosen action
    const nOut = sizes.at(-1);
    act.forEach((name, j) => {
      const cy = y(nOut, j), x0 = x(sizes.length - 1) + 20 * k;
      g.textAlign = 'left'; g.fillStyle = rgba(C.ink, 1); g.font = `500 ${15 * k}px Barlow, sans-serif`;
      g.fillText(name, x0, cy - 9 * k);
      const bw = padR - 40, mid = x0 + bw / 2, v = actions ? Math.max(-1, Math.min(1, actions[j])) : 0;
      g.fillStyle = rgba(C.line, 1); g.fillRect(x0, cy + 1, bw, 8 * k);
      g.fillStyle = rgba(v >= 0 ? C.accent : C.deep, 1);
      g.fillRect(Math.min(mid, mid + v * bw / 2), cy + 1, Math.abs(v) * bw / 2, 8 * k);
      g.fillStyle = rgba(C.ink, 0.6); g.fillRect(mid - 0.5, cy - 1, 1, 12);
    });
    if (title) { g.textAlign = 'left'; g.font = `600 ${15 * k}px "Barlow Condensed", sans-serif`; g.fillStyle = rgba(C.gold, 1); g.fillText(title, 0, H - 2); }
  }
  return { draw, fit };
}

// Weights halfway between two snapshots, for the learning animation.
export function lerpWeights(a, b, t) {
  const l = (p, q) => Array.isArray(p) ? p.map((v, i) => l(v, q[i])) : p + (q - p) * t;
  return { W: l(a.W, b.W), b: l(a.b, b.b) };
}
