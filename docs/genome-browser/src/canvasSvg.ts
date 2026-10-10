/**
 * A 2D context that records what it is asked to draw as SVG.
 *
 * Written for the genome browser's figure export. The obvious alternative -- a second renderer
 * that walks the same sampled columns and emits SVG -- is what this deliberately avoids: two
 * renderers of the same data drift, and the one nobody looks at drifts silently. This records the
 * REAL renderer, so a figure cannot disagree with the screen about anything.
 *
 * It implements only the operations the track canvas actually uses: rects, text, filled `Path2D`
 * glyphs, `moveTo`/`lineTo` strokes, transforms, rectangular clips, dashes and alpha. It carries
 * NO raster path -- `drawImage` is not implemented, and a caller that needs one has a screenshot,
 * not a figure. (The browser's tile decoding uses `drawImage`, but that happens on its own
 * offscreen canvas long before anything is painted.)
 *
 * Geometry is baked per element as a `transform="matrix(...)"` read from the live CTM rather than
 * emitted as nested groups: the transform stack is then correct by construction instead of by a
 * second implementation of save/restore.
 */

interface RecState {
  /** How many `<g clip-path>` wrappers this save level opened, so `restore` closes exactly those. */
  opened: number;
}

const esc = (s: string): string => s.replace(/[&<>"]/g, (c) => (
  c === '&' ? '&amp;' : c === '<' ? '&lt;' : c === '>' ? '&gt;' : '&quot;'));

const num = (n: number): string => (Number.isFinite(n) ? String(Math.round(n * 100) / 100) : '0');

/** A colour the browser may hand us as `rgba(...)`; SVG takes it directly, so only guard nullish. */
const paint = (v: unknown): string => (typeof v === 'string' && v ? v : 'none');

export interface SvgRecorder {
  /** Hand this to the existing renderer in place of a real 2D context. */
  ctx: CanvasRenderingContext2D;
  /** The finished document. Call after the renderer has run. */
  svg(width: number, height: number, background: string, title: string): string;
}

/**
 * @param measure a real 2D context, used ONLY for `measureText` and for the transform stack.
 *   Text metrics have to come from a real engine or every label's width is a guess.
 */
export function createSvgRecorder(measure: CanvasRenderingContext2D): SvgRecorder {
  const body: string[] = [];
  const defs: string[] = [];
  const stack: RecState[] = [];
  let state: RecState = { opened: 0 };
  let clipSeq = 0;
  // The current sub-path, in USER space. Only `moveTo`/`lineTo`/`rect` are used by the callers.
  let path: { x: number; y: number }[] = [];
  let pendingRect: { x: number; y: number; w: number; h: number } | null = null;

  const m = () => {
    const t = measure.getTransform();
    return `matrix(${num(t.a)},${num(t.b)},${num(t.c)},${num(t.d)},${num(t.e)},${num(t.f)})`;
  };
  const common = (): string => {
    const a = measure.globalAlpha;
    return `transform="${m()}"` + (a < 1 ? ` opacity="${num(a)}"` : '');
  };

  const rec = {
    // -- state ------------------------------------------------------------------------------
    save() { stack.push(state); state = { opened: 0 }; measure.save(); },
    restore() {
      for (let i = 0; i < state.opened; i += 1) body.push('</g>');
      state = stack.pop() ?? { opened: 0 };
      measure.restore();
    },
    translate(x: number, y: number) { measure.translate(x, y); },
    scale(x: number, y: number) { measure.scale(x, y); },
    setTransform(...a: unknown[]) { (measure.setTransform as (...z: unknown[]) => void)(...a); },
    getTransform() { return measure.getTransform(); },
    setLineDash(d: number[]) { measure.setLineDash(d); },
    getLineDash() { return measure.getLineDash(); },
    measureText(s: string) { return measure.measureText(s); },
    clearRect() { /* the SVG starts empty; a clear is the background rect the caller supplies */ },

    // -- paths ------------------------------------------------------------------------------
    beginPath() { path = []; pendingRect = null; },
    moveTo(x: number, y: number) { path = [{ x, y }]; },
    lineTo(x: number, y: number) { path.push({ x, y }); },
    closePath() { if (path.length) path.push({ ...path[0] }); },
    rect(x: number, y: number, w: number, h: number) { pendingRect = { x, y, w, h }; },
    roundRect(x: number, y: number, w: number, h: number, r: number | number[]) {
      const rr = Math.max(0, Math.min(typeof r === 'number' ? r : (r[0] ?? 0), w / 2, h / 2));
      pendingRect = { x, y, w, h };
      (pendingRect as { r?: number }).r = rr;
    },
    /**
     * A rectangular clip, emitted in ROOT space and applied through a wrapper `<g>`.
     *
     * This is the one place canvas and SVG genuinely disagree, and getting it wrong is invisible
     * in the file and total on screen. A canvas clip is fixed in DEVICE space the moment `clip()`
     * is called. An SVG `clip-path` is resolved in the user space of the element that REFERENCES
     * it -- so putting it on each glyph, which carries its own translate-and-scale matrix, applies
     * that matrix to the clip rectangle too and the glyph clips itself away. The first version did
     * exactly that: 1,063 valid glyph outlines in the file and three empty lanes on screen.
     *
     * So the rect's corners are pushed through the CTM here, once, and the wrapper carries no
     * transform of its own. Every clip in this renderer is axis-aligned under an axis-aligned CTM,
     * so a corner bounding box is exact rather than an approximation.
     */
    clip() {
      if (!pendingRect) return;
      const t = measure.getTransform();
      const px = (x: number, y: number) => ({ x: t.a * x + t.c * y + t.e, y: t.b * x + t.d * y + t.f });
      const c0 = px(pendingRect.x, pendingRect.y);
      const c1 = px(pendingRect.x + pendingRect.w, pendingRect.y + pendingRect.h);
      const id = `gbclip${clipSeq += 1}`;
      defs.push(`<clipPath id="${id}" clipPathUnits="userSpaceOnUse">`
        + `<rect x="${num(Math.min(c0.x, c1.x))}" y="${num(Math.min(c0.y, c1.y))}" `
        + `width="${num(Math.abs(c1.x - c0.x))}" height="${num(Math.abs(c1.y - c0.y))}"/></clipPath>`);
      body.push(`<g clip-path="url(#${id})">`);
      state.opened += 1;
    },

    // -- painting ---------------------------------------------------------------------------
    fillRect(x: number, y: number, w: number, h: number) {
      if (!(w > 0) || !(h > 0)) return;
      body.push(`<rect x="${num(x)}" y="${num(y)}" width="${num(w)}" height="${num(h)}" `
        + `fill="${paint(measure.fillStyle)}" ${common()}/>`);
    },
    strokeRect(x: number, y: number, w: number, h: number) {
      body.push(`<rect x="${num(x)}" y="${num(y)}" width="${num(w)}" height="${num(h)}" `
        + `fill="none" stroke="${paint(measure.strokeStyle)}" `
        + `stroke-width="${num(measure.lineWidth)}" ${common()}/>`);
    },
    fillText(s: string, x: number, y: number) {
      const text = String(s);
      if (!text) return;
      const anchor = measure.textAlign === 'right' || measure.textAlign === 'end' ? 'end'
        : measure.textAlign === 'center' ? 'middle' : 'start';
      // The canvas font shorthand is CSS, so it goes straight into a `style` and keeps weight,
      // size and family together rather than being taken apart and reassembled.
      body.push(`<text x="${num(x)}" y="${num(y)}" text-anchor="${anchor}" `
        + `fill="${paint(measure.fillStyle)}" style="font:${esc(measure.font)}" `
        + `${common()}>${esc(text)}</text>`);
    },
    /** `fill(path)` is how every DNA glyph is drawn: the outline goes in verbatim. */
    fill(p?: Path2D | string) {
      if (p && typeof p !== 'string' && (p as Path2D & { __d?: string }).__d !== undefined) {
        body.push(`<path d="${esc((p as Path2D & { __d: string }).__d)}" `
          + `fill="${paint(measure.fillStyle)}" ${common()}/>`);
        return;
      }
      if (pendingRect) { rec.fillRect(pendingRect.x, pendingRect.y, pendingRect.w, pendingRect.h); return; }
      if (path.length > 1) {
        body.push(`<path d="${path.map((q, i) => `${i ? 'L' : 'M'}${num(q.x)} ${num(q.y)}`).join('')}Z" `
          + `fill="${paint(measure.fillStyle)}" ${common()}/>`);
      }
    },
    stroke() {
      if (path.length < 2) return;
      const dash = measure.getLineDash();
      body.push(`<path d="${path.map((q, i) => `${i ? 'L' : 'M'}${num(q.x)} ${num(q.y)}`).join('')}" `
        + `fill="none" stroke="${paint(measure.strokeStyle)}" `
        + `stroke-width="${num(measure.lineWidth || 1)}"`
        + (dash.length ? ` stroke-dasharray="${dash.map(num).join(',')}"` : '')
        + ` ${common()}/>`);
    },
  };

  // Everything not recorded above (fillStyle, font, globalAlpha, lineWidth, textAlign, ...) is a
  // plain property and is forwarded to the real context, which is also what `measureText` and the
  // transform stack read. So the recorder holds no duplicate of the drawing state.
  const proxy = new Proxy(measure, {
    get(target, prop, receiver) {
      if (prop in rec) return (rec as Record<string | symbol, unknown>)[prop];
      const v = Reflect.get(target, prop, target);
      return typeof v === 'function' ? (v as (...a: unknown[]) => unknown).bind(target) : v;
    },
    set(target, prop, value) { Reflect.set(target, prop, value, target); return true; },
  }) as unknown as CanvasRenderingContext2D;

  return {
    ctx: proxy,
    svg(width, height, background, title) {
      // A renderer that returns without unwinding its own saves would leave the document
      // unbalanced, and an unbalanced SVG renders as nothing at all.
      let open = state.opened;
      for (const s of stack) open += s.opened;
      const close = '</g>'.repeat(open);
      return `<svg xmlns="http://www.w3.org/2000/svg" width="${num(width)}" `
        + `height="${num(height)}" viewBox="0 0 ${num(width)} ${num(height)}">`
        + `<title>${esc(title)}</title>`
        + (defs.length ? `<defs>${defs.join('')}</defs>` : '')
        + `<rect width="100%" height="100%" fill="${paint(background)}"/>`
        + body.join('')
        + close
        + '</svg>';
    },
  };
}

/**
 * `Path2D` does not expose the string it was built from, and an SVG export needs exactly that.
 *
 * Patching the constructor once, at module load, is what lets the recorder emit a glyph outline
 * verbatim instead of approximating it. It is a no-op for every other user of `Path2D`: the string
 * is stashed on the instance and nothing reads it but the recorder.
 */
export function rememberPathData(): void {
  const g = globalThis as { Path2D?: typeof Path2D; __gbPathPatched?: boolean };
  if (!g.Path2D || g.__gbPathPatched) return;
  g.__gbPathPatched = true;
  const Orig = g.Path2D;
  const Patched = function (this: unknown, ...args: unknown[]) {
    const p = new Orig(...(args as [])) as Path2D & { __d?: string };
    if (typeof args[0] === 'string') p.__d = args[0];
    return p;
  } as unknown as typeof Path2D;
  Patched.prototype = Orig.prototype;
  g.Path2D = Patched;
}
