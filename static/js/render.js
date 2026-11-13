// Draws the hand skeleton and fingertip light trails over the camera.

export const FINGER_HUES = ["#FFB547", "#FF7A59", "#FF4F8B", "#B76BFF", "#5B8CFF"];
const FINGERS = [
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9, 10, 11, 12],
  [13, 14, 15, 16],
  [17, 18, 19, 20],
];
const PALM = [0, 1, 5, 9, 13, 17];
const PALM_BONES = [[0, 1], [0, 5], [5, 9], [9, 13], [13, 17], [0, 17]];
const TRAIL_MS = 420;

export class Overlay {
  constructor(canvas, video) {
    this.canvas = canvas;
    this.video = video;
    this.ctx = canvas.getContext("2d");
    this.mirror = true;
    this.trails = new Map(); // key -> [{x, y, t}]
    this.resize();
    window.addEventListener("resize", () => this.resize());
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.w = this.canvas.clientWidth || window.innerWidth;
    this.h = this.canvas.clientHeight || window.innerHeight;
    this.canvas.width = Math.round(this.w * dpr);
    this.canvas.height = Math.round(this.h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // Normalised video coords -> screen coords, matching object-fit: cover.
  mapper() {
    const vw = this.video.videoWidth || 1280;
    const vh = this.video.videoHeight || 720;
    const s = Math.max(this.w / vw, this.h / vh);
    const dw = vw * s, dh = vh * s;
    const ox = (this.w - dw) / 2, oy = (this.h - dh) / 2;
    const mirror = this.mirror, w = this.w;
    return (p) => {
      let x = ox + p.x * dw;
      if (mirror) x = w - x;
      return { x, y: oy + p.y * dh };
    };
  }

  handBox(result, i = 0) {
    if (!result || !result.landmarks || !result.landmarks[i]) return null;
    const map = this.mapper();
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const p of result.landmarks[i]) {
      const q = map(p);
      x0 = Math.min(x0, q.x); y0 = Math.min(y0, q.y);
      x1 = Math.max(x1, q.x); y1 = Math.max(y1, q.y);
    }
    return { x0, y0, x1, y1 };
  }

  draw(result, now, { trails = true } = {}) {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.w, this.h);
    const hands = (result && result.landmarks) || [];
    const handed = (result && (result.handedness || result.handednesses)) || [];
    const map = this.mapper();
    const seen = new Set();

    hands.forEach((lm, i) => {
      const pts = lm.map(map);
      const key = (handed[i] && handed[i][0] && handed[i][0].categoryName) || `h${i}`;

      // trails: record fingertip positions
      FINGERS.forEach((f, fi) => {
        const k = `${key}:${fi}`;
        seen.add(k);
        let arr = this.trails.get(k);
        if (!arr) { arr = []; this.trails.set(k, arr); }
        const tip = pts[f[3]];
        arr.push({ x: tip.x, y: tip.y, t: now });
      });

      const alpha = 1;
      // palm
      ctx.globalCompositeOperation = "source-over";
      ctx.beginPath();
      PALM.forEach((idx, n) => (n ? ctx.lineTo(pts[idx].x, pts[idx].y) : ctx.moveTo(pts[idx].x, pts[idx].y)));
      ctx.closePath();
      ctx.fillStyle = `rgba(236,239,250,${0.07 * alpha})`;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = `rgba(236,239,250,${0.45 * alpha})`;
      ctx.beginPath();
      for (const [a, b] of PALM_BONES) { ctx.moveTo(pts[a].x, pts[a].y); ctx.lineTo(pts[b].x, pts[b].y); }
      ctx.stroke();

      // fingers, each in its own hue
      FINGERS.forEach((f, fi) => {
        ctx.strokeStyle = hexA(FINGER_HUES[fi], 0.8 * alpha);
        ctx.lineWidth = 2.5;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.beginPath();
        const base = fi === 0 ? pts[1] : pts[f[0]];
        ctx.moveTo(base.x, base.y);
        for (const idx of f) ctx.lineTo(pts[idx].x, pts[idx].y);
        ctx.stroke();
        for (const idx of f.slice(0, 3)) dot(ctx, pts[idx], 2.6, `rgba(236,239,250,${0.85 * alpha})`);
        const tip = pts[f[3]];
        dot(ctx, tip, 9, hexA(FINGER_HUES[fi], 0.22 * alpha));
        dot(ctx, tip, 5, hexA(FINGER_HUES[fi], alpha));
      });
      dot(ctx, pts[0], 3.5, `rgba(236,239,250,${0.9 * alpha})`);
    });

    // trails: fade and draw with additive light
    for (const [k, arr] of this.trails) {
      while (arr.length && now - arr[0].t > TRAIL_MS) arr.shift();
      if (!arr.length && !seen.has(k)) this.trails.delete(k);
    }
    if (!trails) return;
    ctx.globalCompositeOperation = "lighter";
    ctx.lineCap = "round";
    for (const [k, arr] of this.trails) {
      if (arr.length < 2) continue;
      const hue = FINGER_HUES[+k.split(":")[1]];
      for (let pass = 0; pass < 2; pass++) {
        for (let j = 1; j < arr.length; j++) {
          const a = arr[j - 1], b = arr[j];
          const life = 1 - (now - b.t) / TRAIL_MS;
          if (life <= 0) continue;
          ctx.strokeStyle = hexA(hue, pass === 0 ? 0.16 * life : 0.75 * life);
          ctx.lineWidth = pass === 0 ? 4 + 14 * life : 1 + 4 * life;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }
    ctx.globalCompositeOperation = "source-over";
  }
}

function dot(ctx, p, r, fill) {
  ctx.beginPath();
  ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
}

const _hexCache = new Map();
export function hexA(hex, a) {
  let rgb = _hexCache.get(hex);
  if (!rgb) {
    const n = parseInt(hex.slice(1), 16);
    rgb = [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    _hexCache.set(hex, rgb);
  }
  return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${Math.max(0, Math.min(1, a)).toFixed(3)})`;
}

// Intro: a waving hand drawn only as five light trails.
export function introAnimation(canvas) {
  const ctx = canvas.getContext("2d");
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let w, h, raf = 0, stopped = false;
  const hist = FINGER_HUES.map(() => []);
  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = canvas.clientWidth; h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };
  resize();
  window.addEventListener("resize", resize);

  const tips = (t) => {
    const narrow = w < 900;
    const px = narrow ? w * 0.62 : w * 0.7, py = narrow ? h * 1.02 : h * 0.92;
    const R = Math.min(w, h) * (narrow ? 0.5 : 0.56);
    const sway = Math.sin(t * 1.15) * 0.42 + Math.sin(t * 0.37) * 0.08;
    const lens = [0.62, 0.9, 1.0, 0.93, 0.76];
    return FINGER_HUES.map((_, i) => {
      const a = -Math.PI / 2 + sway + (i - 2) * 0.27 + (i === 0 ? -0.22 : 0);
      const curl = 1 - 0.05 * Math.sin(t * 2.3 + i * 0.6);
      return { x: px + Math.cos(a) * R * lens[i] * curl, y: py + Math.sin(a) * R * lens[i] * curl };
    });
  };

  const frame = (ms) => {
    const t = ms / 1000;
    ctx.clearRect(0, 0, w, h);
    const now = tips(t);
    now.forEach((p, i) => {
      hist[i].push({ ...p, t });
      while (hist[i].length && t - hist[i][0].t > 1.6) hist[i].shift();
    });
    ctx.globalCompositeOperation = "lighter";
    ctx.lineCap = "round";
    hist.forEach((arr, i) => {
      for (let pass = 0; pass < 2; pass++) {
        for (let j = 1; j < arr.length; j++) {
          const life = 1 - (t - arr[j].t) / 1.6;
          ctx.strokeStyle = hexA(FINGER_HUES[i], pass ? 0.7 * life : 0.12 * life);
          ctx.lineWidth = pass ? 1 + 3.5 * life : 6 + 16 * life;
          ctx.beginPath();
          ctx.moveTo(arr[j - 1].x, arr[j - 1].y);
          ctx.lineTo(arr[j].x, arr[j].y);
          ctx.stroke();
        }
      }
      const p = arr[arr.length - 1];
      if (p) { dot(ctx, p, 12, hexA(FINGER_HUES[i], 0.25)); dot(ctx, p, 5, FINGER_HUES[i]); }
    });
    ctx.globalCompositeOperation = "source-over";
  };

  if (reduce) {
    for (let ms = 0; ms <= 1600; ms += 16) frame(ms + 2000);
  } else {
    const loop = (ms) => { if (stopped) return; frame(ms); raf = requestAnimationFrame(loop); };
    raf = requestAnimationFrame(loop);
  }
  return () => { stopped = true; cancelAnimationFrame(raf); window.removeEventListener("resize", resize); };
}
