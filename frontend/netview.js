// Draws the policy network: inputs on the left, hidden layers, actions on the right.
// Edge colour shows the sign of each weight, edge strength shows its size.

export function drawNet(canvas, { obs, act, layers, snap, colors }) {
  const r = canvas.getBoundingClientRect(), d = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(r.width * d); canvas.height = Math.round(r.height * d);
  const g = canvas.getContext('2d'); g.setTransform(d, 0, 0, d, 0, 0);
  const w = r.width, h = r.height;
  g.clearRect(0, 0, w, h);
  const MAX = 20;
  // the same evenly spread choice of nodes the server uses for its weight snapshot
  const pick = n => { if (n <= MAX) return [...Array(n).keys()]; const st = (n - 1) / (MAX - 1); return [...new Set([...Array(MAX).keys()].map(i => Math.round(i * st)))]; };
  // the columns of the diagram: sizes and labels
  const sizes = [obs.length, ...layers.map(l => l.units), act.length];
  const shown = sizes.map(n => Math.min(n, MAX));
  const labelL = Math.min(118, w * 0.26), labelR = Math.min(96, w * 0.2);
  const x0 = labelL + 8, x1 = w - labelR - 8, top = 22, bot = h - 18;
  const cx = i => x0 + (sizes.length < 2 ? 0 : i / (sizes.length - 1)) * (x1 - x0);
  const cy = (n, j) => n === 1 ? (top + bot) / 2 : top + j / (n - 1) * (bot - top);
  // edges from the snapshot (it already holds at most MAX rows and columns per layer)
  const L = snap && snap.layers && snap.layers.length === sizes.length - 1 ? snap.layers : null;
  // scale each layer by its own largest weight, so small layers stay readable
  const wmax = L ? L.map(l => Math.max(1e-6, ...l.w.flat().map(Math.abs))) : [];
  for (let i = 0; i < sizes.length - 1; i++) {
    const a = shown[i], b = shown[i + 1];
    for (let q = 0; q < b; q++) for (let p = 0; p < a; p++) {
      let v = 0;
      if (L) { const row = L[i].w[q]; v = row ? row[p] || 0 : 0; }
      const s = L ? Math.min(1, Math.abs(v) / (wmax[i] * 0.7)) : 0.25;
      g.strokeStyle = L ? (v >= 0 ? colors.accent : colors.deep) : colors.line;
      g.globalAlpha = L ? 0.08 + 0.75 * s * s : 0.5;
      g.lineWidth = L ? 0.4 + 1.8 * s : 0.6;
      g.beginPath(); g.moveTo(cx(i), cy(a, p)); g.lineTo(cx(i + 1), cy(b, q)); g.stroke();
    }
  }
  g.globalAlpha = 1;
  // nodes
  const rad = Math.max(2.5, Math.min(5.5, (bot - top) / (Math.max(...shown) * 2.4)));
  sizes.forEach((n, i) => {
    const k = shown[i];
    for (let j = 0; j < k; j++) {
      let bias = 0;
      if (L && i > 0 && L[i - 1].b) bias = L[i - 1].b[j] || 0;
      g.fillStyle = colors.panel; g.strokeStyle = colors.ink; g.lineWidth = 1;
      g.beginPath(); g.arc(cx(i), cy(k, j), rad, 0, 7); g.fill(); g.stroke();
      if (bias) { g.fillStyle = bias > 0 ? colors.accent : colors.deep; g.globalAlpha = Math.min(1, Math.abs(bias) * 3); g.beginPath(); g.arc(cx(i), cy(k, j), rad - 1.5, 0, 7); g.fill(); g.globalAlpha = 1; }
    }
    g.fillStyle = colors.muted; g.font = '500 11px Barlow, sans-serif'; g.textAlign = 'center';
    const name = i === 0 ? 'inputs' : i === sizes.length - 1 ? 'actions' : `${n} ${layers[i - 1].act}`;
    g.fillText(name, cx(i), 12);
    if (n > MAX) g.fillText(`${MAX} of ${n} shown`, cx(i), h - 4);
  });
  // labels for inputs and actions
  g.font = '400 11px Barlow, sans-serif'; g.fillStyle = colors.ink;
  const fitText = (s, max) => { if (g.measureText(s).width <= max) return s; while (s.length > 3 && g.measureText(s + '…').width > max) s = s.slice(0, -1); return s + '…'; };
  g.textAlign = 'right';
  const ins = pick(obs.length);
  for (let j = 0; j < shown[0]; j++) g.fillText(fitText(obs[ins[j]] || '', labelL - 4), x0 - rad - 6, cy(shown[0], j) + 4);
  g.textAlign = 'left';
  const std = snap && snap.std && snap.std.length === act.length ? snap.std : null;
  for (let j = 0; j < shown.at(-1); j++) {
    const label = (act[j] || '') + (std ? ` ±${std[j].toFixed(2)}` : '');
    g.fillText(fitText(label, labelR - 4), x1 + rad + 6, cy(shown.at(-1), j) + 4);
  }
}
