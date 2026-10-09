// The 3D stage: draws replays of episodes that the Python trainer sends over.
import * as THREE from './vendor/three.module.min.js';

const G = 9.81, rnd = Math.random, clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const GHOST = { transparent: true, opacity: 0.3, depthWrite: false };
const KID = { skin: 0xF2C9A8, hair: 0x4A2E22, tee: 0x5B7FB5, jeans: 0x3F5E94, shoe: 0x3B4A8C, rim: 0x9AA3A8, eye: 0x3A241C };
const FURS = [0xC8915A, 0x8A5A33, 0xE2C08E, 0x5E4630];
const MAXG = 20; // actors per game (1 shown + up to 19 ghosts)
const hex = c => '#' + c.toString(16).padStart(6, '0').toUpperCase();
const TIME = { key: 'time', label: 'Time of day', type: 'select', options: [['day', 'Day'], ['sunset', 'Sunset'], ['night', 'Night']] };
// Cosmetic options per game: they change the picture, never the physics. Only the main actor is dressed; ghosts stay plain.
// A swatch's first option is the default; null there means the theme colour named by `token`.
export const LOOKS = {
  diver: [
    { key: 'suit', label: 'Swimsuit', type: 'swatch', token: 'accent', options: [null, '#2B59D9', '#1E9E5A', '#E8B90E', '#B03BD1', '#1B1B1F'] },
    { key: 'skin', label: 'Skin', type: 'swatch', options: [hex(KID.skin), '#E0AC84', '#B97A56', '#8D5524', '#5C3A21'] },
    { key: 'hair', label: 'Hair', type: 'swatch', options: [hex(KID.hair), '#17110E', '#D6B05E', '#B5532C', '#A3A3A3'] },
    { key: 'cap', label: 'Swim cap', type: 'check', def: false },
    { key: 'glasses', label: 'Glasses', type: 'check', def: true },
    { key: 'pool', label: 'Water', type: 'select', options: [['', 'Pool blue'], ['#27C9B8', 'Tropical'], ['#4E9B6B', 'Lagoon'], ['#2D3F8F', 'Deep sea']] },
    TIME,
  ],
  dog: [
    { key: 'fur', label: 'Fur', type: 'swatch', options: [...FURS.map(hex), '#F1ECE2', '#242022'] },
    { key: 'collar', label: 'Collar', type: 'swatch', token: 'accent', options: [null, '#2B59D9', '#1E9E5A', '#F04FA0', '#1B1B1F'] },
    { key: 'hat', label: 'Party hat', type: 'check', def: false },
    { key: 'frisbee', label: 'Frisbee', type: 'swatch', token: 'accent', options: [null, '#F2D21B', '#2BB3E0', '#7ED957', '#F04FA0'] },
    { key: 'shirt', label: "Thrower's shirt", type: 'swatch', options: [hex(KID.tee), '#D9481F', '#2E8B57', '#F2C230', '#7A4FB5'] },
    TIME,
  ],
  pizza: [
    { key: 'outfit', label: "Chef's clothes", type: 'swatch', options: ['#F7F7F2', '#1F1F24', '#C8322A', '#2E6FB5', '#3E8E4E'] },
    { key: 'hat', label: 'Chef hat', type: 'check', def: true },
    { key: 'toppings', label: 'Toppings', type: 'select', options: [['', 'Plain dough'], ['margherita', 'Margherita'], ['pepperoni', 'Pepperoni'], ['olives', 'Olives']] },
    { key: 'wall', label: 'Kitchen wall', type: 'swatch', token: 'wall', options: [null, '#F3E1C7', '#CDE7D3', '#F4C9C1', '#34404D'] },
  ],
  walker: [
    { key: 'shirt', label: 'Shirt', type: 'swatch', token: 'accent', options: [null, '#2B59D9', '#1E9E5A', '#E8B90E', '#B03BD1', '#1B1B1F'] },
    { key: 'pants', label: 'Trousers', type: 'swatch', options: [hex(KID.jeans), '#2A2A2E', '#8A6A4A', '#C8322A', '#E7E2D6'] },
    { key: 'skin', label: 'Skin', type: 'swatch', options: [hex(KID.skin), '#E0AC84', '#B97A56', '#8D5524', '#5C3A21'] },
    { key: 'hair', label: 'Hair', type: 'swatch', options: [hex(KID.hair), '#17110E', '#D6B05E', '#B5532C', '#A3A3A3'] },
    { key: 'glasses', label: 'Glasses', type: 'check', def: true },
    { key: 'ground', label: 'Ground', type: 'select', options: [['', 'Grass'], ['#D9C48A', 'Desert'], ['#EEF3F6', 'Snow'], ['#8C8C8C', 'Moon rock']] },
    TIME,
  ],
  sandwich: [
    { key: 'table', label: 'Table', type: 'select', options: [['', 'White board'], ['wood', 'Wooden table'], ['picnic', 'Picnic cloth']] },
    { key: 'cheese', label: 'Cheese', type: 'select', options: [['', 'Cheddar'], ['swiss', 'Swiss']] },
    { key: 'toast', label: 'Toasted bread', type: 'check', def: false },
  ],
};
export const lookDefaults = key => Object.fromEntries(LOOKS[key].map(o => [o.key, o.type === 'check' ? o.def : o.type === 'select' ? o.options[0][0] : o.options[0]]));
// lighting for the time of day; day uses the theme's sky
const TIMES = {
  day: { sun: 0xffffff, si: 0.72, sy: 14, hemi: 0xffffff, ground: 0x6b8a94, hi: 0.62 },
  sunset: { sky: '#F2A97E', sun: 0xFFB070, si: 0.62, sy: 4, hemi: 0xFFD2B0, ground: 0x7A5A6A, hi: 0.5 },
  night: { sky: '#0E1838', sun: 0x9DB2FF, si: 0.22, sy: 14, hemi: 0x6F82B8, ground: 0x1A2236, hi: 0.3 },
};
const TOAST = 'sepia(0.3) saturate(1.7) brightness(0.8)';

// `now` (optional) returns the time in seconds for idle animation such as waves; the video renderer passes its own clock.
// `onEvent(episode, index, kind)` (optional) hears every landing, catch or splash as it plays, e.g. to log sounds.
export function createStage({ canvas, flat, tags, onNoGL, now = () => performance.now() / 1000, gl = {}, onEvent = null }) {
  let W = 0, H = 0, renderer = null, col = {}, game = null, games = {};
  let disp = [], clock = 0, showAll = false, compare = false, parts = [];
  const scene = new THREE.Scene(), cam = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
  const hemi = new THREE.HemisphereLight(0xffffff, 0x6b8a94, 0.62 * Math.PI); scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xffffff, 0.72 * Math.PI);
  sun.position.set(7, 14, 9); sun.target.position.set(3, 0, 0); scene.add(sun, sun.target);
  sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048); sun.shadow.bias = -0.002; sun.shadow.normalBias = 0.03;
  Object.assign(sun.shadow.camera, { left: -11, right: 13, top: 10, bottom: -9, near: 1, far: 45 });
  const backdrop = new THREE.Group(); scene.add(backdrop);
  const lam = (c, o) => new THREE.MeshLambertMaterial(Object.assign({ color: c }, o || {}));
  const tinted = [];
  const tm = (token, o) => { const m = lam(0xffffff, o); tinted.push([m, token]); return m; };
  // a material coloured by a look option: [material, game, option, fallback colour or theme token]
  const looks = Object.fromEntries(Object.keys(LOOKS).map(k => [k, lookDefaults(k)])), looked = [];
  const lm = (gk, key, fallback, o) => { const m = lam(0xffffff, o); looked.push([m, gk, key, fallback]); return m; };
  const paintLooks = () => looked.forEach(([m, gk, k, fb]) => m.color.set(looks[gk][k] || col[fb] || fb));
  const blk = (g, w, h, d, m, x, y, z) => { const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m); b.position.set(x, y, z); g.add(b); return b; };
  const limb = (r, h, m) => { const g = new THREE.Group(), c = new THREE.Mesh(new THREE.CylinderGeometry(r, r * 0.85, h, 10), m); c.position.y = -h / 2; g.add(c); return g; };
  const vis = i => i === 0 || showAll || (compare && i === 1 && disp.hasFirst);

  // returns groups for the hair, glasses and hat, so a look can switch them on and off
  function kidHead(parent, y, g, hat, mats = {}) {
    const skin = mats.skin || lam(KID.skin, g), hair = mats.hair || lam(KID.hair, g), rim = lam(KID.rim, g), eye = lam(KID.eye, g), R = 0.24;
    const head = { hair: new THREE.Group(), glasses: new THREE.Group(), hat: new THREE.Group() };
    Object.values(head).forEach(p => parent.add(p));
    const add = (geo, m, x, yy, z, to = parent) => { const q = new THREE.Mesh(geo, m); q.position.set(x, y + yy, z || 0); to.add(q); return q; };
    add(new THREE.SphereGeometry(R, 20, 16), skin, 0, 0);
    for (let k = 0; k < 26; k++) {
      const v = (k + 0.5) / 26, ph = Math.acos(1 - 1.25 * v), th = k * 2.39996, x = Math.sin(ph) * Math.cos(th), yy = Math.cos(ph), z = Math.sin(ph) * Math.sin(th);
      if (x > 0.55 && yy < 0.62) continue;
      add(new THREE.SphereGeometry(0.085 + 0.02 * ((k * 7) % 3), 10, 8), hair, x * R * 0.98, yy * R * 0.98, z * R * 0.98, head.hair);
    }
    [1, -1].forEach(sd => {
      add(new THREE.SphereGeometry(0.05, 10, 8), skin, -0.02, -0.03, sd * 0.245);
      add(new THREE.SphereGeometry(0.03, 10, 8), eye, 0.215, 0, sd * 0.095);
      const ring = add(new THREE.TorusGeometry(0.075, 0.011, 8, 22), rim, 0.228, -0.005, sd * 0.095, head.glasses); ring.rotation.y = Math.PI / 2;
      add(new THREE.BoxGeometry(0.2, 0.014, 0.014), rim, 0.11, 0.0, sd * 0.2, head.glasses);
    });
    add(new THREE.BoxGeometry(0.014, 0.014, 0.06), rim, 0.232, 0.0, 0, head.glasses);
    add(new THREE.SphereGeometry(0.03, 8, 8), skin, 0.245, -0.05, 0);
    if (hat) {
      const w = lam(0xFAFAF6, g);
      add(new THREE.CylinderGeometry(0.2, 0.19, 0.2, 16), w, -0.02, 0.26, 0, head.hat);
      add(new THREE.SphereGeometry(0.24, 14, 10), w, -0.02, 0.42, 0, head.hat).scale.y = 0.6;
    }
    return head;
  }
  // o.top / o.leg: materials for the clothes (default: a tee and jeans, or chef's whites with o.hat)
  function person(o) {
    const g = o.ghost ? GHOST : {}, skin = lam(KID.skin, g), top = o.top || lam(o.hat ? 0xF7F7F2 : KID.tee, g), leg = o.leg || lam(o.hat ? 0xF7F7F2 : KID.jeans, g), shoe = lam(KID.shoe, g), white = lam(0xFFFFFF, g);
    const root = new THREE.Group(), body = new THREE.Group(); body.position.y = 0.15; root.add(body);
    const add = (geo, m, x, y) => { const p = new THREE.Mesh(geo, m); p.position.set(x, y, 0); body.add(p); };
    add(new THREE.CylinderGeometry(0.19, 0.17, 0.5, 14), top, 0, 0.24); add(new THREE.CylinderGeometry(0.17, 0.17, 0.2, 14), leg, 0, -0.1);
    const head = kidHead(body, 0.74, g, o.hat);
    const arms = [1, -1].map(sd => { const a = limb(0.055, 0.5, skin); a.position.set(0, 0.46, sd * 0.24); const sl = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.07, 0.2, 10), top); sl.position.y = -0.08; a.add(sl); body.add(a); return a; });
    [1, -1].forEach(sd => {
      const h = limb(0.085, 0.42, leg); h.position.set(0, -0.2, sd * 0.09); const knee = limb(0.075, 0.42, leg); knee.position.y = -0.42;
      const f = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.09, 0.12), shoe); f.position.set(0.05, -0.44, 0); knee.add(f);
      const so = new THREE.Mesh(new THREE.BoxGeometry(0.25, 0.03, 0.125), white); so.position.set(0.05, -0.495, 0); knee.add(so); h.add(knee); body.add(h);
    });
    return { root, arms, head };
  }
  function dogModel(i) {
    const g = i ? GHOST : {}, fur = i ? lam(FURS[i % 4], g) : lm('dog', 'fur', hex(FURS[0])), dark = lam(0x3A2A1E, g), root = new THREE.Group();
    const add = (geo, m, x, y, z, rz) => { const p = new THREE.Mesh(geo, m); p.position.set(x, y, z || 0); if (rz) p.rotation.z = rz; root.add(p); return p; };
    add(new THREE.CylinderGeometry(0.17, 0.17, 0.62, 12), fur, 0, 0, 0, Math.PI / 2);
    add(new THREE.SphereGeometry(0.17, 12, 10), fur, 0.31, 0); add(new THREE.SphereGeometry(0.17, 12, 10), fur, -0.31, 0);
    add(new THREE.SphereGeometry(0.17, 14, 12), fur, 0.47, 0.2);
    add(new THREE.BoxGeometry(0.22, 0.11, 0.13), fur, 0.64, 0.15);
    add(new THREE.SphereGeometry(0.04, 8, 8), dark, 0.75, 0.18);
    [1, -1].forEach(s => { add(new THREE.BoxGeometry(0.07, 0.2, 0.1), dark, 0.4, 0.3, s * 0.14, 0.3); add(new THREE.SphereGeometry(0.028, 8, 8), dark, 0.58, 0.27, s * 0.1); });
    if (!i) {
      const collar = lm('dog', 'collar', 'accent');
      add(new THREE.CylinderGeometry(0.185, 0.185, 0.05, 14), collar, 0.36, 0.08, 0, Math.PI / 2 - 0.4);
      const hat = add(new THREE.ConeGeometry(0.075, 0.22, 14), collar, 0.44, 0.45, 0, 0.3), pom = new THREE.Mesh(new THREE.SphereGeometry(0.032, 8, 8), lam(0xFAFAF6));
      pom.position.y = 0.11; hat.add(pom); root.userData.hat = hat;
    }
    const legs = [[0.27, 0.1], [0.27, -0.1], [-0.27, 0.1], [-0.27, -0.1]].map(([x, z]) => { const l = limb(0.05, 0.34, fur); l.position.set(x, -0.1, z); root.add(l); return l; });
    const tail = limb(0.035, 0.3, fur); tail.position.set(-0.42, 0.08, 0); tail.rotation.z = -2.4; root.add(tail);
    return { root, legs, tail };
  }
  function ragModel(i) {
    const g = i ? GHOST : {}, skin = i ? lam(KID.skin, g) : lm('diver', 'skin', hex(KID.skin)), suit = i ? lam(KID.jeans, g) : lm('diver', 'suit', 'accent'), root = new THREE.Group();
    root.userData.suit = suit;
    const ps = [0, 1, 2, 3].map(() => { const p = new THREE.Group(); root.add(p); return p; });
    const add = (k, geo, m, x, y, z) => { const q = new THREE.Mesh(geo, m); q.position.set(x, y, z || 0); ps[k].add(q); return q; };
    add(0, new THREE.CylinderGeometry(0.16, 0.17, 0.26, 14), suit, 0, -0.3); add(0, new THREE.CylinderGeometry(0.19, 0.16, 0.46, 14), skin, 0, 0.04);
    root.userData.head = kidHead(ps[0], 0.44, g, false, { skin, hair: i ? null : lm('diver', 'hair', hex(KID.hair)) });
    if (!i) { const cap = add(0, new THREE.SphereGeometry(0.257, 20, 10, 0, Math.PI * 2, 0, Math.PI * 0.56), suit, 0, 0.44); cap.rotation.z = 0.35; root.userData.cap = cap; }
    [1, -1].forEach(sd => {
      add(1, new THREE.CylinderGeometry(0.08, 0.07, 0.46, 10), skin, 0, 0, sd * 0.09); add(1, new THREE.CylinderGeometry(0.095, 0.09, 0.2, 10), suit, 0, 0.13, sd * 0.09);
      add(2, new THREE.CylinderGeometry(0.062, 0.052, 0.46, 10), skin, 0, 0, sd * 0.09); add(2, new THREE.BoxGeometry(0.2, 0.06, 0.09), skin, 0.06, -0.25, sd * 0.09);
      add(3, new THREE.CylinderGeometry(0.048, 0.052, 0.62, 10), skin, 0, 0, sd * 0.24);
    });
    return { root, parts: ps };
  }

  // ----- splash particles and sounds -----
  const MAXP = 600, pGeo = new THREE.BufferGeometry(), pPos = new Float32Array(MAXP * 3);
  pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
  const dc = document.createElement('canvas'); dc.width = dc.height = 32; const dg = dc.getContext('2d'); dg.fillStyle = '#fff'; dg.beginPath(); dg.arc(16, 16, 14, 0, 7); dg.fill();
  const mDrop = new THREE.PointsMaterial({ color: 0xffffff, size: 0.3, map: new THREE.CanvasTexture(dc), transparent: true, alphaTest: 0.4 });
  const points = new THREE.Points(pGeo, mDrop); points.frustumCulled = false; scene.add(points);
  function burst(x, y, z, n, pw, token, up) {
    pw = Math.min(pw, 5.5);
    mDrop.color.set(col[token]);
    for (let j = 0; j < n && parts.length < MAXP; j++) parts.push({ x: x + (rnd() - .5) * 0.4, y, z: z + (rnd() - .5) * 0.4, vx: (rnd() - .5) * pw, vy: (up ? 0.3 + rnd() * 0.7 : rnd() - .5) * pw * 1.6, vz: (rnd() - .5) * pw * 0.7, life: 1.4 });
  }
  let actx = null, noise = null, lastSnd = 0;
  const sound = {
    on: false,
    toggle() {
      if (!actx) {
        try {
          actx = new (window.AudioContext || window.webkitAudioContext)();
          noise = actx.createBuffer(1, actx.sampleRate * 1.5, actx.sampleRate);
          const d = noise.getChannelData(0); for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
        } catch (e) { return null; }
      }
      this.on = !this.on;
      if (this.on) actx.resume(); else actx.suspend();
      return this.on;
    },
  };
  const ready = () => actx && actx.state === 'running' && performance.now() - lastSnd >= 140 && (lastSnd = performance.now());
  function hiss(sp, type) {
    if (!ready()) return;
    const t = actx.currentTime, src = actx.createBufferSource(), f = actx.createBiquadFilter(), g = actx.createGain();
    const vol = Math.min(1, Math.pow(0.03 + sp / 55, 1.3)), len = type === 'highpass' ? 0.18 + Math.min(1.1, sp * 0.02) : 0.2 + Math.min(1, sp * 0.014);
    src.buffer = noise; f.type = type; f.frequency.value = type === 'highpass' ? 2600 : Math.max(500, 2600 - sp * 32);
    g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(Math.max(0.001, vol), t + 0.01); g.gain.exponentialRampToValueAtTime(0.0001, t + len);
    src.connect(f); f.connect(g); g.connect(actx.destination); src.start(t); src.stop(t + len + 0.05);
  }
  const boom = sp => hiss(sp, 'lowpass'), tish = sp => hiss(sp, 'highpass');
  function ding() {
    if (!ready()) return;
    const t = actx.currentTime, o = actx.createOscillator(), g = actx.createGain();
    o.type = 'sine'; o.frequency.setValueAtTime(880, t); o.frequency.exponentialRampToValueAtTime(1320, t + 0.12);
    g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.25, t + 0.01); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.3);
    o.connect(g); g.connect(actx.destination); o.start(t); o.stop(t + 0.32);
  }
  // linear interpolation into a flat frame array with `n` floats per frame
  function lerpFrames(fr, n, t) {
    const count = fr.length / n, f = clamp(t * 60, 0, count - 1), i0 = Math.floor(f), i1 = Math.min(count - 1, i0 + 1), u = f - i0;
    return k => fr[i0 * n + k] + (fr[i1 * n + k] - fr[i0 * n + k]) * u;
  }

  // ----- games -----
  games.diver = {
    cam: [3.4, 4.4, 16, 2.6, 2.5, 0],
    prep(s) { s.ev = [{ t: s.E, k: 'land' }]; },
    // pull the camera back for high boards, high jumps and long flights; ordinary dives keep the original framing
    frame(s) {
      const up = Math.max(0, (s.board || 0) - 3, s.m.height - 7), right = Math.max(0, Math.max(s.m.distance, s.px) - 8), f = Math.max(up, right * 0.6);
      cam.position.set(3.4 + 0.5 * right, 4.4 + 0.75 * up, 16 + 2.1 * f); cam.lookAt(2.6 + 0.5 * right, 2.5 + 0.45 * up, 0);
    },
    look(v) {
      this.act.forEach((d, i) => { const u = d.root.userData; u.head.glasses.visible = v.glasses; if (!i) { u.cap.visible = v.cap; u.head.hair.visible = !v.cap; } });
    },
    tip(s, t) { const u = t - s.wt; return u < 0 ? -s.amp * Math.sin(Math.PI * t / s.wt) : 0.8 * s.amp * Math.exp(-3 * u) * Math.sin(17 * u); },
    build() {
      const g = this.group = new THREE.Group(), deck = tm('deck');
      const wm = new THREE.MeshPhongMaterial({ color: 0xffffff, specular: 0xffffff, shininess: 90, transparent: true, opacity: 0.78, depthWrite: false }); looked.push([wm, 'diver', 'pool', 'water']);
      this.water = new THREE.Mesh(new THREE.PlaneGeometry(30, 10, 104, 34), wm); this.water.rotation.x = -Math.PI / 2; this.water.position.set(13.5, 0, 0); g.add(this.water);
      blk(g, 30, 0.2, 10, tm('deep'), 13.5, -2.7, 0); blk(g, 8, 2.9, 10, deck, -5.5, -1.2, 0); blk(g, 40, 3.4, 2, deck, 10.5, -0.95, -6);
      this.pillar = blk(g, 0.6, 1, 0.9, deck, -2.6, 1.55, 0);
      this.board = new THREE.Group(); this.board.position.set(-3.05, 2.88, 0); g.add(this.board); blk(this.board, 3.2, 0.08, 1.5, tm('board'), 1.6, 0, 0);
      this.act = Array.from({ length: MAXG }, (_, i) => { const d = ragModel(i); d.root.position.z = i ? ((i * 7) % 20 - 9.5) * 0.06 : 0; g.add(d.root); return d; });
      this.rings = [0, 1, 2].map(() => { const r = new THREE.Mesh(new THREE.RingGeometry(0.82, 1, 40), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, side: THREE.DoubleSide, depthWrite: false })); r.rotation.x = -Math.PI / 2; r.position.y = 0.04; r.visible = false; g.add(r); return r; });
      const pm = new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: 0.12, gapSize: 0.1 }); tinted.push([pm, 'ink']);
      this.peak = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-0.8, 0, 0), new THREE.Vector3(0.8, 0, 0)]), pm); this.peak.computeLineDistances(); g.add(this.peak);
    },
    update(sims, t) {
      this.act.forEach((d, i) => {
        const s = sims[i]; d.root.visible = !!s && vis(i); if (!d.root.visible) return;
        const L = lerpFrames(s.fr, 12, t);
        d.parts.forEach((q, k) => { q.position.x = L(k * 3); q.position.y = L(k * 3 + 1); q.rotation.z = L(k * 3 + 2); });
      });
      const b = sims[0], top = b.board - 0.12, h = Math.max(0.2, top - 0.25);
      this.board.position.y = top; this.board.rotation.z = Math.atan2(this.tip(b, t), 3.05);
      this.pillar.scale.y = h; this.pillar.position.y = 0.25 + h / 2;
      const x = b.m.distance;
      this.rings.forEach((r, j) => { const tt = t - b.E - j * 0.2; r.visible = tt > 0 && tt < 1.3; if (r.visible) { const sc = 0.35 + tt * (0.9 + b.m.splash * 0.07); r.scale.set(sc, sc, 1); r.position.x = x; r.material.opacity = (1 - tt / 1.3) * 0.95; } });
      const P = this.water.geometry.attributes.position, nowS = now(), age = t - b.E, A = age > 0 ? Math.min(0.45, 0.05 + b.m.splash * 0.008) * Math.exp(-age * 1.5) : 0;
      for (let k = 0; k < P.count; k++) {
        const px = P.getX(k) + 13.5, py = P.getY(k); let hh = 0.035 * Math.sin(px * 1.3 + nowS * 1.6) + 0.025 * Math.sin(py * 1.9 + nowS * 1.2);
        if (A) { const d = Math.hypot(px - x, py); if (d < 2.4 * age + 0.6) hh += A * Math.sin(7 * (d - 2.4 * age)) / (1 + d * d * 0.6); }
        P.setZ(k, hh);
      }
      P.needsUpdate = true; this.water.geometry.computeVertexNormals();
      this.peak.visible = t < b.E; this.peak.position.set(b.px, b.m.height, 0);
    },
    event(s, i) { if (i === 0) tish(s.m.splash); if (vis(i)) burst(s.m.distance, 0.05, this.act[i].root.position.z, i ? Math.min(8, s.m.splash * 0.18) : Math.min(240, 8 + s.m.splash * 4), 2.4 + s.m.splash * 0.16, 'drop', true); },
    labels(b, t) {
      const L = [[-2.2, b.board + 0.1, b.board.toFixed(1) + ' m springboard']];
      if (t < b.E) L.push([b.px + 1.9, b.m.height - 0.1, 'Peak ' + b.m.height.toFixed(1) + ' m']);
      else L.push([b.m.distance, 2.1, 'Splash ' + b.m.splash.toFixed(0) + ' L, entry ' + b.m.entry_angle.toFixed(0) + '° off vertical']);
      if (b.g && Math.abs(b.g - G) > 0.05) L.push([b.m.distance + 3, 0.6, 'Gravity ' + b.g.toFixed(1) + ' m/s²']);
      return L;
    },
    stat: m => m.splash.toFixed(1) + ' L',
  };

  const fx = (s, t) => -0.6 + s.fvx * t, fy = (s, t) => Math.max(0.03, 1.6 + s.fvy * t - s.fa * t * t);
  games.dog = {
    cam: [3.1, 2.3, 12.5, 3.1, 1.8, 0],
    prep(s) { const c = s.tc >= 0; s.ev = [{ t: s.tl, k: 'land' }, c ? { t: s.tc, k: 'catch' } : { t: s.tf, k: 'drop' }]; },
    // zoom out to keep long, high throws and big jumps in the picture
    frame(s) {
      let dy = 0; for (let k = 1; k < s.fr.length; k += 3) dy = Math.max(dy, s.fr[k]);
      const xmax = Math.max(7.7, fx(s, s.tf) + 1, s.xl + 1.2), ymax = Math.max(5.6, 2.2 + s.fvy * s.fvy / (4 * s.fa), dy + 0.8);
      const k = Math.max((xmax + 1.5) / 9.2, ymax / 5.6);
      cam.position.set(-1.5 + 4.6 * k, 2.3 + 2.8 * (k - 1), 12.5 * k); cam.lookAt(-1.5 + 4.6 * k, 1.8 + 2.5 * (k - 1), 0);
    },
    look(v) { this.act[0].root.userData.hat.visible = v.hat; },
    build() {
      const g = this.group = new THREE.Group();
      blk(g, 120, 0.2, 16, tm('grass'), 20, -0.1, 0);
      const th = person({ top: lm('dog', 'shirt', hex(KID.tee)) }); th.root.position.set(-1.1, 0.98, 0); th.arms[0].rotation.z = 1.8; g.add(th.root);
      this.act = Array.from({ length: MAXG }, (_, i) => { const d = dogModel(i); d.root.position.z = i ? ((i * 7) % 20 - 9.5) * 0.09 : 0; g.add(d.root); return d; });
      [-0.6, 4.6, 8.4].forEach((x, j) => {
        const tr = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.18, 1.2, 8), tm('board')); tr.position.set(x, 0.6, -5 - j % 2); g.add(tr);
        const top = new THREE.Mesh(new THREE.ConeGeometry(1.1, 2.6, 10), lam(0x3F7F4A)); top.position.set(x, 2.4, -5 - j % 2); g.add(top);
      });
      this.fris = new THREE.Mesh(new THREE.CylinderGeometry(0.17, 0.17, 0.035, 18), lm('dog', 'frisbee', 'accent')); g.add(this.fris);
    },
    update(sims, t) {
      this.act.forEach((d, i) => {
        const s = sims[i]; d.root.visible = !!s && vis(i); if (!d.root.visible) return;
        const L = lerpFrames(s.fr, 3, t), bx = L(0), by = L(1), ang = L(2), c = Math.cos(ang), sn = Math.sin(ang);
        d.root.position.x = bx - 0.07 * sn; d.root.position.y = by + 0.07 * c; d.root.rotation.z = ang;
        const sw = t > s.tj && t < s.tl ? 0.7 : t > s.w && t <= s.tj ? Math.sin(bx * 7) * 0.8 : 0;
        d.legs[0].rotation.z = d.legs[3].rotation.z = sw; d.legs[1].rotation.z = d.legs[2].rotation.z = -sw;
        d.tail.rotation.z = -2.4 + (s.m.caught && t > s.tc ? Math.sin(t * 25) * 0.5 : 0);
        if (i === 0) {
          if (s.m.caught && t >= s.tc) { this.fris.position.set(bx + 0.7 * c - 0.22 * sn, by + 0.7 * sn + 0.22 * c, 0); this.fris.rotation.z = ang; }
          else { const ft = Math.min(t, s.tf); this.fris.position.set(fx(s, ft), fy(s, ft), 0); this.fris.rotation.z = 0; }
        }
      });
    },
    event(s, i, k) {
      if (k === 'catch') { if (i === 0) { ding(); burst(fx(s, s.tc), fy(s, s.tc), 0, 30, 2.5, 'accent', false); } }
      else if (k === 'drop') { if (i === 0) boom(6); }
      else if (!s.m.landing) { if (i === 0) boom(24); if (vis(i)) burst(s.xl, 0.1, this.act[i].root.position.z, i ? 6 : 40, 2.2, 'board', true); }
    },
    labels(b, t) {
      const L = [];
      if (t >= b.E) L.push(b.m.caught ? [fx(b, b.tc), fy(b, b.tc) + 1.3, 'Caught at ' + b.m.catch_height.toFixed(1) + ' m'] : [fx(b, b.tf), 1.0, 'Missed by ' + b.m.miss.toFixed(1) + ' m']);
      if (b.m.jumped && t >= b.tl && Math.abs(b.m.spins) > 0.4) L.push([b.xl, 1.6, b.m.landing ? Math.abs(b.m.spins).toFixed(0) + ' spin landing' : 'Bad landing']);
      return L;
    },
    stat: m => m.caught ? m.catch_height.toFixed(1) + ' m' : 'Miss',
  };

  const X0 = 0.6, H0 = 1.7;
  const qa = new THREE.Quaternion(), qb = new THREE.Quaternion(), ZA = new THREE.Vector3(0, 0, 1), YA = new THREE.Vector3(0, 1, 0);
  games.pizza = {
    cam: [0.9, 6.6, 8.6, 0.7, 1.9, 0],
    prep(s) { s.ev = [{ t: s.E, k: 'end' }]; },
    // tilt up and back for high ceilings
    frame(s) { const f = Math.max(0, s.ceil - 4.3); cam.position.set(0.9, 6.6 + 0.9 * f, 8.6 + 1.6 * f); cam.lookAt(0.7, 1.9 + 0.55 * f, 0); },
    look(v) { this.chef.head.hat.visible = v.hat; Object.entries(this.tops).forEach(([k, grp]) => { grp.visible = k === v.toppings; }); },
    // topping layers that sit on the main dough and stretch and spin with it
    toppings(d) {
      const T = {}, sauce = lam(0xC8432B), mozz = lam(0xFBF6E6), basil = lam(0x3F8F3A), pep = lam(0xA8322A), olive = lam(0x26221F);
      const disc = (grp, r, h, m, x, z, y) => { const q = new THREE.Mesh(new THREE.CylinderGeometry(r, r, h, 18), m); q.position.set(x, y, z); grp.add(q); return q; };
      const ring = (n, rr, ph = 0) => Array.from({ length: n }, (_, k) => [Math.cos(k / n * Math.PI * 2 + ph) * rr, Math.sin(k / n * Math.PI * 2 + ph) * rr]);
      ['margherita', 'pepperoni', 'olives'].forEach(k => { const grp = T[k] = new THREE.Group(); grp.visible = false; d.add(grp); disc(grp, 0.8, 0.012, sauce, 0, 0, 0.03); });
      [...ring(5, 0.5), [0, 0]].forEach(([x, z]) => disc(T.margherita, 0.17, 0.014, mozz, x, z, 0.04));
      ring(5, 0.36, 0.63).forEach(([x, z]) => { disc(T.margherita, 0.075, 0.01, basil, x, z, 0.045).scale.z = 0.55; });
      [...ring(6, 0.52), ...ring(3, 0.2, 0.5)].forEach(([x, z]) => disc(T.pepperoni, 0.14, 0.016, pep, x, z, 0.042));
      [...ring(8, 0.55), ...ring(4, 0.25, 0.4)].forEach(([x, z]) => { const o = new THREE.Mesh(new THREE.TorusGeometry(0.055, 0.024, 6, 14), olive); o.rotation.x = Math.PI / 2; o.position.set(x, 0.045, z); T.olives.add(o); });
      return T;
    },
    build() {
      const g = this.group = new THREE.Group(), wall = lm('pizza', 'wall', 'wall');
      blk(g, 20, 0.2, 8, tm('board'), 0, -0.1, 0); blk(g, 20, 9, 0.3, wall, 0, 3.9, -3);
      this.ceil = blk(g, 7, 0.12, 5, tm('deck', { transparent: true, opacity: 0.3, depthWrite: false }), 0.6, 3.81, -0.4);
      blk(g, 3.4, 0.9, 0.8, tm('deck'), 0.2, 0.45, -2.2);
      const ow = lm('pizza', 'outfit', '#F7F7F2');
      this.chef = person({ hat: true, top: ow, leg: ow }); this.chef.root.position.set(0, 0.98, 0); g.add(this.chef.root);
      this.act = Array.from({ length: MAXG }, (_, i) => {
        const d = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 0.05, 28), tm('dough', i ? GHOST : {}));
        const bump = new THREE.Mesh(new THREE.SphereGeometry(0.09, 8, 8), tm('board', i ? GHOST : {})); bump.position.set(0.82, 0.02, 0); d.add(bump);
        const hole = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.34, 0.07, 16), wall); hole.position.set(0.2, 0, 0.1); d.add(hole); d.hole = hole; g.add(d); return d;
      });
      this.tops = this.toppings(this.act[0]);
    },
    update(sims, t) {
      this.act.forEach((d, i) => {
        const s = sims[i]; d.visible = !!s && vis(i); if (!d.visible) return;
        const L = lerpFrames(s.fr, 6, t);
        d.position.set(L(0), L(1), 0);
        qa.setFromAxisAngle(ZA, L(2)); qb.setFromAxisAngle(YA, L(3)); d.quaternion.copy(qa).multiply(qb);
        const r = 0.12 + (s.r - 0.12) * Math.min(1, t / Math.max(0.01, s.air)); d.scale.set(r, 1, r * (0.55 + 0.45 * s.m.roundness));
        d.hole.visible = !!s.m.torn && t > s.air * 0.6;
      });
      const b = sims[0], L = lerpFrames(b.fr, 6, t), hx = L(4), hy = L(5);
      this.chef.root.position.x = hx - X0;
      const lift = (hy - (H0 - 0.06)) * 1.6, sw = 1.7 + lift + 0.7 * Math.exp(-t * 7);
      this.chef.arms.forEach(a => { a.rotation.z = sw; });
      this.ceil.position.y = b.ceil + 0.06;
    },
    event(s, i) {
      const mess = s.m.dropped * 35 + s.m.ceiling * 28 + s.m.torn * 12; if (i === 0) boom(3 + mess);
      if ((s.m.dropped || s.m.ceiling) && vis(i)) burst(s.xE, s.m.ceiling ? s.ceil - 0.2 : 0.1, 0, i ? 5 : 50, 2, 'dough', !s.m.ceiling);
    },
    labels(b, t) {
      const L = [[-1.5, b.ceil - 0.25, 'Ceiling ' + b.ceil.toFixed(1) + ' m'], [3.2, 0.5, 'Draught ' + (b.wind > 0 ? '→ ' : '← ') + Math.abs(b.wind).toFixed(1)]];
      if (t >= b.E) L.push([b.xE, b.m.dropped ? 0.9 : b.m.ceiling ? b.ceil - 0.6 : H0 + 0.9, b.m.ceiling ? 'Stuck to the ceiling' : b.m.dropped ? 'Dropped on the floor' : (b.m.torn ? 'Torn, ' : '') + b.m.size.toFixed(0) + ' cm, ' + (b.m.roundness * 100).toFixed(0) + '% round' + (b.m.caught ? ', caught' : '')]);
      return L;
    },
    stat: m => m.ceiling ? 'Stuck' : m.dropped ? 'Dropped' : m.torn ? 'Torn' : m.size.toFixed(0) + ' cm',
  };

  let breadBase = null, breadCut = null;
  games.sandwich = {
    flat: true,
    prep(s) { s.ev = [0, 1, 2, 3, 4].map(j => ({ t: 0.2 + j * 0.5 + 0.33, k: j < 4 ? 'pat' : 'end' })); },
    build() { this.group = new THREE.Group(); this.c = flat.getContext('2d'); },
    table() {
      const c = this.c, v = looks.sandwich.table;
      if (v === 'wood') {
        c.fillStyle = '#C99A62'; c.fillRect(0, 0, W, H);
        const n = 6, ph = H / n;
        for (let k = 0; k < n; k++) {
          const y = k * ph; c.fillStyle = k % 2 ? 'rgba(120,70,30,0.10)' : 'rgba(255,230,190,0.10)'; c.fillRect(0, y, W, ph);
          c.strokeStyle = 'rgba(110,62,24,0.22)'; c.lineWidth = 1;
          for (let j = 1; j < 4; j++) { c.beginPath(); for (let x = 0; x <= W; x += 16) c.lineTo(x, y + j * ph / 4 + 2.5 * Math.sin(x / 70 + k * 3 + j)); c.stroke(); }
          c.fillStyle = 'rgba(80,44,16,0.45)'; c.fillRect(0, y, W, 1.5);
        }
      } else if (v === 'picnic') {
        const s = Math.max(24, Math.min(W, H) / 9);
        c.fillStyle = '#FBF7F0'; c.fillRect(0, 0, W, H); c.fillStyle = 'rgba(214,64,58,0.4)';
        for (let x = 0; x < W; x += s) c.fillRect(x, 0, s / 2, H);
        for (let y = 0; y < H; y += s) c.fillRect(0, y, W, s / 2);
      } else { c.fillStyle = '#FEFEFE'; c.fillRect(0, 0, W, H); }
    },
    piece(j) {
      const c = this.c, blob = (R, amp, n, ph) => { c.beginPath(); for (let k = 0; k <= 48; k++) { const a = k / 48 * Math.PI * 2, r = R * (1 + amp * Math.sin(a * n + ph)); c.lineTo(Math.cos(a) * r, Math.sin(a) * r); } c.closePath(); };
      const out = (fill, stroke) => { c.fillStyle = fill; c.fill(); c.shadowColor = 'transparent'; c.strokeStyle = stroke; c.lineWidth = 0.014; c.lineJoin = 'round'; c.stroke(); };
      if (j === 0) { blob(0.42, 0.07, 9, 0); out('#9BD065', '#3F7D2C'); c.strokeStyle = '#6FB145'; c.lineWidth = 0.012; for (let k = 0; k < 5; k++) { const a = k * 1.26 + 0.3; c.beginPath(); c.moveTo(0, 0); c.lineTo(Math.cos(a) * 0.3, Math.sin(a) * 0.3); c.stroke(); } }
      else if (j === 1) { blob(0.36, 0.05, 3, 1); out('#F4A6AE', '#B4566B'); blob(0.24, 0.05, 3, 1); c.fillStyle = '#F8C3C8'; c.fill(); }
      else if (j === 2 && looks.sandwich.cheese === 'swiss') {
        c.beginPath(); c.rect(-0.3, -0.3, 0.6, 0.6); out('#F7E7A6', '#B99A3A'); c.fillStyle = '#E8CF78';
        [[-0.15, -0.14, 0.07], [0.12, -0.08, 0.05], [0.16, 0.16, 0.08], [-0.1, 0.13, 0.045], [0.02, 0.02, 0.035], [-0.22, 0.02, 0.03]].forEach(h => { c.beginPath(); c.arc(h[0], h[1], h[2], 0, 7); c.fill(); });
      }
      else if (j === 2) { c.beginPath(); c.rect(-0.3, -0.3, 0.6, 0.6); out('#F8CC4A', '#B5830F'); c.fillStyle = '#E3AC2B'; [[-0.12, -0.1, 0.06], [0.13, 0.05, 0.05], [-0.02, 0.16, 0.04]].forEach(h => { c.beginPath(); c.arc(h[0], h[1], h[2], 0, 7); c.fill(); }); }
      else if (j === 3) {
        [-1, 1].forEach(sg => {
          c.save(); c.translate(sg * 0.2, 0); c.beginPath(); c.arc(0, 0, 0.19, 0, 7); out('#E64A3B', '#98231A'); c.beginPath(); c.arc(0, 0, 0.13, 0, 7); c.fillStyle = '#F27C63'; c.fill();
          c.fillStyle = '#FBE3B0'; for (let k = 0; k < 6; k++) { c.beginPath(); c.arc(Math.cos(k * 1.05) * 0.08, Math.sin(k * 1.05) * 0.08, 0.018, 0, 7); c.fill(); } c.restore();
        });
      }
      else if (breadCut) { if (looks.sandwich.toast) c.filter = TOAST; c.drawImage(breadCut, -0.8945, -0.8813, 1.789, 1.789); c.filter = 'none'; }
      else { c.beginPath(); c.rect(-0.5, -0.55, 1, 1.1); out(looks.sandwich.toast ? '#D9A35E' : '#F3DDA6', '#7A4A1C'); }
    },
    scene(s, t, cx, cy, S, anim) {
      const c = this.c, U = 0.559 * S, by = cy - 0.0074 * S, d = Math.min(2, window.devicePixelRatio || 1);
      const ox = s.bx * U, oy = s.by * U;
      // the photo sits on white: on another table, or toasted, use the cut-out
      const toast = looks.sandwich.toast, im = (toast || looks.sandwich.table) && breadCut ? breadCut : breadBase;
      if (im) { if (toast) c.filter = TOAST; c.drawImage(im, cx - S / 2 + ox, cy - S / 2 + oy, S, S); c.filter = 'none'; }
      else { c.fillStyle = toast ? '#D9A35E' : '#F3DDA6'; c.fillRect(cx + ox - 0.5 * U, by + oy - 0.55 * U, U, 1.1 * U); }
      s.it.forEach((it, j) => {
        const p = anim ? clamp((t - 0.2 - j * 0.5) / 0.33, 0, 1) : 1; if (p <= 0) return;
        const e = 1 - (1 - p) * (1 - p), k = 1 + 0.6 * (1 - e), sh = (0.035 + 0.14 * (1 - e)) * U * d;
        c.save(); c.globalAlpha = Math.min(1, p * 4); c.translate(cx + it[0] * U, by + it[1] * U); c.rotate(it[2]); c.scale(U * k, U * k);
        c.shadowColor = 'rgba(140,105,50,0.32)'; c.shadowOffsetX = sh; c.shadowOffsetY = sh; c.shadowBlur = 3 * d; this.piece(j); c.restore();
      });
    },
    update(sims, t) {
      const c = this.c; this.table();
      const b = sims[0]; this.scene(b, t, W / 2, H / 2 - 8, Math.min(W, H) * 0.9, true);
      c.font = '500 14px Barlow, sans-serif'; c.fillStyle = '#4A3520';
      if (sims[1] && vis(1) && compare) {
        const S = Math.min(W, H) * 0.3; c.strokeStyle = '#D9CDB8'; c.lineWidth = 1; c.strokeRect(8.5, 8.5, S, S);
        c.save(); c.beginPath(); c.rect(9, 9, S - 1, S - 1); c.clip(); this.scene(sims[1], 99, 8 + S / 2, 8 + S / 2, S, false); c.restore();
        c.textAlign = 'left'; c.fillStyle = '#4A3520'; c.fillText('First try', 12, S + 26);
      }
      if (t >= b.E) { c.textAlign = 'center'; c.fillText('Covered ' + b.m.coverage.toFixed(0) + '%, ' + b.m.overhang.toFixed(0) + '% hanging out, lid ' + b.m.lid_offset.toFixed(1) + ' cm off and turned ' + b.m.lid_twist.toFixed(0) + '°', W / 2, H - 10); }
    },
    event(s, i, k) { if (i === 0) boom(k === 'pat' ? 2 : 3 + s.m.overhang * 0.5 + s.m.lid_offset * 3 + s.m.lid_twist * 0.15); },
    labels() { return []; },
    stat: m => m.coverage.toFixed(0) + '%',
  };

  // ----- walker: a two-legged ragdoll on rough ground, with crates, wind and shoves -----
  // frame layout: (x, y, angle) for torso, left thigh, left shin, right thigh, right shin
  const WK = { torso: [0.34, 0.62], thigh: [0.13, 0.46], shin: [0.11, 0.46], foot: [0.28, 0.07] };
  games.walker = {
    solo: true,
    prep(s) {
      s.ev = (s.events || []).filter(e => e.k === 'shove').map(e => ({ t: e.t, k: 'shove', v: e.v }));
      if (s.m && s.m.fell) s.ev.push({ t: s.E, k: 'fall' });
      s.ev.forEach(e => { e.done = false; });
    },
    build() {
      const g = this.group = new THREE.Group();
      const shirt = lm('walker', 'shirt', 'accent'), pants = lm('walker', 'pants', hex(KID.jeans)), skin = lm('walker', 'skin', hex(KID.skin));
      const far = lam(0x2F4A78), shoe = lam(KID.shoe);
      this.groundTop = lm('walker', 'ground', 'grass'); this.groundSide = lam(0x7A5A3A);
      this.terrainMesh = null;
      const part = () => { const p = new THREE.Group(); g.add(p); return p; };
      const box = (p, w, h, d, m, x = 0, y = 0) => { const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m); b.position.set(x, y, 0); p.add(b); return b; };
      const torso = part();
      box(torso, WK.torso[0], WK.torso[1], 0.3, shirt);
      this.head = kidHead(torso, WK.torso[1] / 2 + 0.27, {}, false, { skin, hair: lm('walker', 'hair', hex(KID.hair)) });
      // arms hang from the shoulders and swing with the legs (for looks only)
      this.arms = [1, -1].map(sd => { const a = limb(0.05, 0.52, skin); a.position.set(0, WK.torso[1] / 2 - 0.06, sd * 0.2); const sl = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.065, 0.18, 10), shirt); sl.position.y = -0.07; a.add(sl); torso.add(a); return a; });
      const legs = [0, 1].map(k => {
        const m = k ? far : pants, z = k ? -0.1 : 0.1;
        const thigh = part(); box(thigh, WK.thigh[0], WK.thigh[1], 0.13, m).position.z = z;
        const shin = part(); box(shin, WK.shin[0], WK.shin[1], 0.11, m).position.z = z;
        box(shin, WK.foot[0], WK.foot[1], 0.12, shoe, 0.07, -WK.shin[1] / 2 - WK.foot[1] / 2).position.z = z;
        return [thigh, shin];
      });
      this.parts = [torso, legs[0][0], legs[0][1], legs[1][0], legs[1][1]];
      this.crates = [];
      this.crateMat = tm('board');
      this.arrow = new THREE.Group();
      const am = tm('accent'); const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.7, 10), am); shaft.rotation.z = Math.PI / 2; this.arrow.add(shaft);
      const tip = new THREE.Mesh(new THREE.ConeGeometry(0.13, 0.28, 12), am); tip.rotation.z = -Math.PI / 2; tip.position.x = 0.48; this.arrow.add(tip);
      this.arrow.visible = false; g.add(this.arrow);
      const wg = new THREE.BufferGeometry(); this.windPos = new Float32Array(90 * 3); wg.setAttribute('position', new THREE.BufferAttribute(this.windPos, 3));
      this.windSeed = Array.from({ length: 90 }, () => [rnd(), rnd() * 4, (rnd() - 0.5) * 3]);
      this.wind = new THREE.Points(wg, new THREE.PointsMaterial({ color: 0xffffff, size: 0.07, transparent: true, opacity: 0.85 })); this.wind.frustumCulled = false; g.add(this.wind);
      this.camX = 0; this.camY = 1;
      this.act = [];
    },
    look(v) { this.head.glasses.visible = !!v.glasses; },
    // the ground, rebuilt whenever an episode brings a new course
    setTerrain(pts) {
      if (this.terrainPts === pts) return;
      this.terrainPts = pts;
      if (this.terrainMesh) { this.group.remove(this.terrainMesh); this.terrainMesh.geometry.dispose(); }
      const sh = new THREE.Shape();
      sh.moveTo(pts[0], -4);
      for (let i = 0; i < pts.length; i += 2) sh.lineTo(pts[i], pts[i + 1]);
      sh.lineTo(pts[pts.length - 2], -4); sh.closePath();
      const geo = new THREE.ExtrudeGeometry(sh, { depth: 3, bevelEnabled: false });
      geo.translate(0, 0, -1.5);
      this.terrainMesh = new THREE.Mesh(geo, [this.groundSide, this.groundTop]);
      this.terrainMesh.receiveShadow = true;
      this.group.add(this.terrainMesh);
    },
    groundAt(x) {
      const p = this.terrainPts; if (!p) return 0;
      for (let i = 2; i < p.length; i += 2) if (p[i] >= x) { const u = (x - p[i - 2]) / (p[i] - p[i - 2] || 1); return p[i - 1] + (p[i + 1] - p[i - 1]) * u; }
      return p[p.length - 1];
    },
    frameAt(s, t) { const n = s.fr.length / 15; return Math.max(0, Math.min(n - 1, t * 60)); },
    windAt(s, t) { let w = 0; (s.winds || []).forEach(([wt, f]) => { if (wt <= t) w = f; }); return w; },
    update(sims, t) {
      const s = sims[0];
      if (!s || !s.fr.length) return;
      this.setTerrain(s.terrain);
      const L = lerpFrames(s.fr, 15, t);
      this.parts.forEach((p, k) => { p.position.set(L(k * 3), L(k * 3 + 1), 0); p.rotation.z = L(k * 3 + 2); p.visible = true; });
      // arms swing opposite to the legs
      const swing = (L(5) - L(11)) * 0.6;
      this.arms[0].rotation.z = -swing - 0.1; this.arms[1].rotation.z = swing - 0.1;
      // crates: those born by this frame, in order
      const f = Math.round(this.frameAt(s, t)), cf = (s.cfr && s.cfr[Math.min(f, s.cfr.length - 1)]) || [], list = s.crates || [];
      while (this.crates.length < list.length) { const c = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), this.crateMat); c.castShadow = c.receiveShadow = true; this.group.add(c); this.crates.push(c); }
      this.crates.forEach((c, j) => {
        const alive = j < list.length && j * 3 + 2 < cf.length;
        c.visible = alive; if (!alive) return;
        const z = list[j].size; c.scale.set(z, z, z * 1.6); c.position.set(cf[j * 3], cf[j * 3 + 1], 0); c.rotation.z = cf[j * 3 + 2];
      });
      // camera follows the torso
      const tx = L(0), ty = this.groundAt(tx);
      this.camX += (tx - this.camX) * 0.12; this.camY += (ty - this.camY) * 0.08;
      cam.position.set(this.camX + 1.0, this.camY + 2.0, 7.2); cam.lookAt(this.camX + 1.0, this.camY + 0.9, 0);
      sun.position.set(this.camX + 7, 14, 9); sun.target.position.set(this.camX + 3, 0, 0);
      // shove arrow, for half a second after each shove
      const sh = (s.events || []).filter(e => e.k === 'shove' && t >= e.t && t < e.t + 0.5).pop();
      this.arrow.visible = !!sh;
      if (sh) { const d = Math.sign(sh.v) || 1; this.arrow.position.set(tx - d * 0.75, L(1) + 0.15, 0.3); this.arrow.rotation.y = d > 0 ? 0 : Math.PI; }
      // wind streaks drifting across the view
      const w = this.windAt(s, t); this.wind.visible = Math.abs(w) > 5;
      if (this.wind.visible) {
        const now_ = now(), sp = w / 40;
        this.windSeed.forEach(([u, h, z], i) => { const span = 14, x = this.camX - 6 + ((u * span + now_ * sp) % span + span) % span; this.windPos[i * 3] = x; this.windPos[i * 3 + 1] = this.camY + 0.2 + h; this.windPos[i * 3 + 2] = z; });
        this.wind.geometry.attributes.position.needsUpdate = true;
      }
    },
    event(s, i, k) {
      const L = lerpFrames(s.fr, 15, clock);
      if (k === 'shove') boom(10);
      else if (k === 'fall') { boom(26); burst(L(0), this.groundAt(L(0)) + 0.1, 0, 40, 2.2, 'board', true); }
    },
    labels(s, t) {
      if (!s || !s.fr.length) return [];
      const L = lerpFrames(s.fr, 15, t), x = L(0), y = L(1), out = [];
      const d = x - (s.fr[0] || 0);
      out.push([x, y + 1.05, (s.m && s.m.fell && t >= s.E) ? `Fell after ${d.toFixed(1)} m` : `${d.toFixed(1)} m`]);
      const w = this.windAt(s, t); if (Math.abs(w) > 5) out.push([this.camX + 2.8, this.camY + 3.2, `Wind ${w > 0 ? '→' : '←'} ${Math.abs(w).toFixed(0)} N`]);
      const sh = (s.events || []).filter(e => e.k === 'shove' && t >= e.t && t < e.t + 0.8).pop(); if (sh) out.push([x - 0.4, y + 1.55, 'Shove!']);
      return out;
    },
    stat: m => m.fell ? `${m.distance.toFixed(1)} m, fell` : `${m.distance.toFixed(1)} m`,
  };

  // ----- setup -----
  Object.values(games).forEach(g => { g.build(); g.group.visible = false; scene.add(g.group); });
  [[-1, 7.6, -9, 1.5], [6, 8.8, -11, 2], [11, 6.9, -9, 1.3], [3, 6.2, -12, 1.1]].forEach(([x, y, z, r]) => {
    [0, 1, 2].forEach(j => { const c = new THREE.Mesh(new THREE.SphereGeometry(r * (j === 1 ? 0.6 : 0.42), 14, 10), tm('panel')); c.position.set(x + (j - 1) * r * 0.62, y + (j === 1 ? r * 0.12 : 0), z); c.scale.y = 0.62; backdrop.add(c); });
  });
  // night sky: a moon and stars behind the clouds
  const night = new THREE.Group(); night.visible = false; backdrop.add(night);
  const moon = new THREE.Mesh(new THREE.SphereGeometry(0.9, 20, 14), new THREE.MeshBasicMaterial({ color: 0xF4F1DE })); moon.position.set(9.5, 9.5, -16); night.add(moon);
  const sGeo = new THREE.BufferGeometry(), sPos = new Float32Array(160 * 3);
  for (let k = 0; k < 160; k++) { sPos[k * 3] = -20 + rnd() * 50; sPos[k * 3 + 1] = 3 + rnd() * 18; sPos[k * 3 + 2] = -18; }
  sGeo.setAttribute('position', new THREE.BufferAttribute(sPos, 3));
  night.add(new THREE.Points(sGeo, new THREE.PointsMaterial({ color: 0xffffff, size: 0.09 })));
  const keyOf = g => Object.keys(games).find(k => games[k] === g);
  function applyTime() {
    const t = game && looks[keyOf(game)].time, T = TIMES[t] || TIMES.day;
    scene.background = new THREE.Color(T.sky || col.sky || '#DDF0F4');
    sun.color.set(T.sun); sun.intensity = T.si * Math.PI; sun.position.y = T.sy;
    hemi.color.set(T.hemi); hemi.groundColor.set(T.ground); hemi.intensity = T.hi * Math.PI;
    night.visible = t === 'night';
  }
  Object.entries(games).forEach(([k, g]) => g.look && g.look(looks[k]));
  Object.values(games).forEach(g => g.group.traverse(o => { if (o.isMesh) { o.castShadow = !o.material.transparent; o.receiveShadow = true; } }));
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, ...gl });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  } catch (e) { onNoGL && onNoGL(); }

  function prepBread(im) {
    const n = im.naturalWidth; if (!n) return; breadBase = im;
    try {
      const cc = document.createElement('canvas'); cc.width = cc.height = n; const x = cc.getContext('2d'); x.drawImage(im, 0, 0);
      const d = x.getImageData(0, 0, n, n), p = d.data, seen = new Uint8Array(n * n), st = [];
      for (let k = 0; k < n; k++) st.push(k, (n - 1) * n + k, k * n, k * n + n - 1);
      while (st.length) {
        const i = st.pop(); if (seen[i] || Math.min(p[i * 4], p[i * 4 + 1], p[i * 4 + 2]) <= 186) continue; seen[i] = 1; p[i * 4 + 3] = 0;
        const X = i % n; if (X > 0) st.push(i - 1); if (X < n - 1) st.push(i + 1); if (i >= n) st.push(i - n); if (i < n * (n - 1)) st.push(i + n);
      }
      x.putImageData(d, 0, 0); breadCut = cc;
    } catch (e) { breadCut = null; }
  }
  const bread = new Image(); bread.onload = () => { prepBread(bread); draw(); }; bread.src = 'assets/bread.jpg';

  function readColors() {
    const s = getComputedStyle(document.documentElement);
    ['sky', 'water', 'deep', 'board', 'deck', 'accent', 'ink', 'muted', 'line', 'panel', 'drop', 'grass', 'dough', 'wall'].forEach(n => col[n] = s.getPropertyValue('--' + n).trim());
    tinted.forEach(([m, t]) => m.color.set(col[t])); paintLooks(); applyTime();
  }
  function fit(c) { const r = c.getBoundingClientRect(), d = Math.min(2, window.devicePixelRatio || 1); c.width = Math.round(r.width * d); c.height = Math.round(r.height * d); c.getContext('2d').setTransform(d, 0, 0, d, 0, 0); return r; }
  function resize() {
    if (!game) return;
    const r = (game.flat ? flat : canvas).getBoundingClientRect(); W = r.width; H = r.height;
    if (game.flat) fit(flat);
    if (renderer) renderer.setSize(W, H, false);
    cam.aspect = W / Math.max(1, H); cam.updateProjectionMatrix();
  }
  const pv = new THREE.Vector3();
  function draw() {
    if (!game) return;
    parts.forEach((p, i) => { pPos[i * 3] = p.x; pPos[i * 3 + 1] = p.y; pPos[i * 3 + 2] = p.z; });
    pGeo.setDrawRange(0, parts.length); pGeo.attributes.position.needsUpdate = true;
    if (!disp.length) {
      tags.forEach(el => { el.hidden = true; });
      if (game.flat) game.table();
      else { Object.values(games).forEach(g => { if (g.act) g.act.forEach(a => { (a.root || a).visible = false; }); }); if (renderer) renderer.render(scene, cam); }
      return;
    }
    game.update(disp, clock);
    const L = game.labels(disp[0], clock);
    tags.forEach((el, i) => {
      const l = L[i]; el.hidden = !l; if (!l) return; if (el.textContent !== l[2]) el.textContent = l[2];
      pv.set(l[0], l[1], 0).project(cam); const hw = el.offsetWidth / 2 + 4;
      el.style.transform = 'translate(' + clamp((pv.x + 1) / 2 * W, hw, Math.max(hw, W - hw)) + 'px,' + Math.max(24, (1 - pv.y) / 2 * H) + 'px) translate(-50%,-100%)';
    });
    if (renderer && !game.flat) renderer.render(scene, cam);
  }
  function endTime() {
    if (!disp.length) return 0;
    let T = disp[0].T;
    disp.forEach((s, i) => { if (vis(i)) T = Math.max(T, s.T); });
    return T;
  }

  return {
    games,
    sound,
    setGame(key) {
      if (game) game.group.visible = false;
      game = games[key]; game.group.visible = true;
      sun.position.set(7, 14, 9); sun.target.position.set(3, 0, 0);  // the walker moves the sun with its camera backdrop.visible = key !== 'pizza'; applyTime();
      canvas.hidden = !!game.flat; flat.hidden = !game.flat;
      disp = []; parts = []; clock = 0;
      resize();
      if (game.cam) { cam.position.set(game.cam[0], game.cam[1], game.cam[2]); cam.lookAt(game.cam[3], game.cam[4], game.cam[5]); }
      draw();
    },
    // eps: episodes sorted best first; first: best episode of the very first iteration
    load(eps, first) {
      const list = eps.slice(0, MAXG);
      if (first && list.length) list.splice(1, 0, first);
      disp = list.slice(0, MAXG).map(s => { const c = { ...s }; game.prep(c); return c; });
      disp.hasFirst = !!first;
      clock = 0; parts = [];
      if (game.frame && disp.length) game.frame(disp[0]);
    },
    clear() { disp = []; parts = []; draw(); },
    // advance playback; returns true once the replay has finished
    tick(dt) {
      if (!disp.length) { draw(); return true; }
      let left = dt;
      while (left > 0) {
        const step = Math.min(left, 0.03); left -= step; clock += step;
        parts.forEach(p => { p.x += p.vx * step; p.y += p.vy * step; p.z += p.vz * step; p.vy -= G * step; p.life -= step * 1.1; });
        parts = parts.filter(p => p.life > 0 && p.y > -0.1);
        disp.forEach((s, i) => s.ev.forEach(e => { if (!e.done && clock >= e.t) { e.done = true; if (vis(i)) { game.event(s, i, e.k); if (onEvent) onEvent(s, i, e.k); } } }));
      }
      draw();
      return clock > endTime();
    },
    setShowAll(v) { showAll = v; draw(); },
    // ----- live play: the server streams an episode while it happens -----
    liveReset(r) {
      const ep = { fr: [], cfr: [], crates: [], events: [], winds: [], terrain: r.terrain, m: { distance: 0, fell: 0 }, E: 1e9, T: 1e9, live: true };
      game.prep(ep); disp = [ep]; disp.hasFirst = false; clock = 0; parts = [];
      if (game.camX !== undefined) { game.camX = 0; game.camY = 1; }
    },
    liveChunk(c) {
      const ep = disp[0]; if (!ep || !ep.live) return;
      ep.fr.push(...c.fr); ep.cfr.push(...c.cfr); ep.crates = c.crates; ep.m = c.m;
      c.events.forEach(e => { ep.events.push(e); if (e.k === 'shove') ep.ev.push({ t: e.t, k: 'shove', v: e.v }); });
      ep.winds.push(...c.winds);
    },
    liveEnd(m) { const ep = disp[0]; if (!ep || !ep.live) return; ep.m = m; ep.E = ep.fr.length / 15 / 60; if (m.fell) ep.ev.push({ t: ep.E, k: 'fall' }); },
    // live playback: keep just behind the newest frame
    liveTick(dt) {
      const ep = disp[0]; if (!ep || !ep.live) return;
      const have = ep.fr.length / 15 / 60;
      let target = Math.min(clock + dt, have);
      if (have - target > 0.4) target = have - 0.1;  // fell behind: catch up
      this.tick(Math.max(0, target - clock));
    },
    // where a click on the stage lands, on the plane of the game (z = 0)
    pick(cx, cy) {
      const r = canvas.getBoundingClientRect(), v = new THREE.Vector3(((cx - r.left) / r.width) * 2 - 1, -((cy - r.top) / r.height) * 2 + 1, 0.5).unproject(cam);
      const dir = v.sub(cam.position).normalize(), k = -cam.position.z / dir.z;
      return { x: cam.position.x + dir.x * k, y: cam.position.y + dir.y * k };
    },
    // dress a game's scene; `v` holds LOOKS options, missing ones fall back to their defaults
    setLook(key, v) {
      looks[key] = { ...lookDefaults(key), ...v };
      paintLooks(); if (games[key].look) games[key].look(looks[key]);
      if (game === games[key]) applyTime();
      draw();
    },
    get clock() { return clock; },
    // colour a ghost's suit (diver only); null restores the default
    tint(i, color) { const a = games.diver.act[i]; if (a && a.root.userData.suit) a.root.userData.suit.color.set(color ?? KID.jeans); },
    // a world point in CSS pixels on the stage
    project(x, y, z = 0) { pv.set(x, y, z).project(cam); return { x: (pv.x + 1) / 2 * W, y: (1 - pv.y) / 2 * H }; },
    depth(i) { const a = game.act; return a && a[i] ? (a[i].root || a[i]).position.z : 0; },
    setCamera(p, look) { cam.position.set(...p); cam.lookAt(...look); draw(); },
    setCompare(v) { compare = v; draw(); },
    stat(key, m) { return games[key].stat(m); },
    resize() { resize(); draw(); },
    readColors() { readColors(); draw(); },
  };
}
