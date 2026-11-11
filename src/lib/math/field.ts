/**
 * Sampled loss field, log-normalised colour mapping and marching-squares isolines.
 * Runs once per (surface, resolution, theme) change, never per frame.
 */
import type { Surface } from './surfaces.ts';

export interface Field {
	nx: number;
	ny: number;
	/** Normalised log-loss in [0, 1], row-major, row 0 = yMax (top of the image). */
	t: Float32Array;
	fMin: number;
	fMax: number;
	/** Offset δ in log(f − fMin + δ). */
	delta: number;
}

/** Decades of dynamic range shown by the colour map. */
export const LOG_DECADES = 5;

export function sampleField(s: Surface, nx: number, ny: number): Field {
	const { xMin, xMax, yMin, yMax } = s.domain;
	const raw = new Float64Array(nx * ny);
	let fMin = Infinity;
	let fMax = -Infinity;
	for (let j = 0; j < ny; j++) {
		const y = yMax - ((yMax - yMin) * j) / (ny - 1);
		for (let i = 0; i < nx; i++) {
			const x = xMin + ((xMax - xMin) * i) / (nx - 1);
			const f = s.f(x, y);
			raw[j * nx + i] = f;
			if (f < fMin) fMin = f;
			if (f > fMax) fMax = f;
		}
	}
	if (s.minima.length) fMin = Math.min(fMin, s.fRef);
	const range = Math.max(fMax - fMin, 1e-12);
	const delta = range * 10 ** -LOG_DECADES;
	const lo = Math.log(delta);
	const hi = Math.log(range + delta);
	const t = new Float32Array(nx * ny);
	for (let k = 0; k < raw.length; k++) t[k] = (Math.log(raw[k] - fMin + delta) - lo) / (hi - lo);
	return { nx, ny, t, fMin, fMax, delta };
}

/** Map a loss value to the same normalised scale as the field (for legends/readouts). */
export function normalise(field: Field, f: number): number {
	const range = Math.max(field.fMax - field.fMin, 1e-12);
	const lo = Math.log(field.delta);
	const hi = Math.log(range + field.delta);
	return (Math.log(Math.max(f - field.fMin, 0) + field.delta) - lo) / (hi - lo);
}

// ---------- colour ----------

/** OKLab → sRGB bytes (clamped). */
export function oklabToRgb(L: number, a: number, b: number): [number, number, number] {
	const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
	const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
	const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
	const lin = [
		4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
		-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
		-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s
	];
	return lin.map((c) => {
		const v = c <= 0.0031308 ? 12.92 * c : 1.055 * Math.max(c, 0) ** (1 / 2.4) - 0.055;
		return Math.round(255 * Math.min(1, Math.max(0, v)));
	}) as [number, number, number];
}

function oklchToLab(L: number, C: number, hDeg: number): [number, number, number] {
	const h = (hDeg * Math.PI) / 180;
	return [L, C * Math.cos(h), C * Math.sin(h)];
}

function oklchToRgb(L: number, C: number, hDeg: number): [number, number, number] {
	return oklabToRgb(...oklchToLab(L, C, hDeg));
}

export interface Ramp {
	/** 256 × RGB lookup table. */
	lut: Uint8ClampedArray;
	/** Isoline stroke colour. */
	iso: string;
}

/**
 * Single-hue sequential ramp. Low loss recedes toward the page surface in both themes
 * (light: near-white; dark: near-surface), high loss is the strongest step.
 */
export function makeRamp(theme: 'light' | 'dark'): Ramp {
	const [L0, L1, C0, C1, H] =
		theme === 'light' ? [0.98, 0.66, 0.006, 0.045, 255] : [0.205, 0.53, 0.012, 0.055, 255];
	const lut = new Uint8ClampedArray(256 * 3);
	for (let i = 0; i < 256; i++) {
		const u = i / 255;
		const [r, g, b] = oklchToRgb(L0 + (L1 - L0) * u, C0 + (C1 - C0) * u, H);
		lut[3 * i] = r;
		lut[3 * i + 1] = g;
		lut[3 * i + 2] = b;
	}
	return {
		lut,
		iso: theme === 'light' ? 'rgba(25, 35, 60, 0.14)' : 'rgba(215, 225, 255, 0.13)'
	};
}

/**
 * Diverging ramp for probabilities: class 0 (blue) ← neutral (p = 0.5) → class 1 (red).
 * Each arm is interpolated linearly in OKLab; poles are light tints so data points stay on top.
 */
export function makeDivergingRamp(theme: 'light' | 'dark'): Ramp {
	const [Ln, Lp, C] = theme === 'light' ? [0.965, 0.78, 0.085] : [0.235, 0.42, 0.09];
	const neutral = oklchToLab(Ln, 0.004, 90);
	const blue = oklchToLab(Lp, C, 255);
	const red = oklchToLab(Lp, C, 25);
	const lut = new Uint8ClampedArray(256 * 3);
	for (let i = 0; i < 256; i++) {
		const u = i / 255;
		const [pole, w] = u < 0.5 ? [blue, 1 - 2 * u] : [red, 2 * u - 1];
		const [r, g, b] = oklabToRgb(
			neutral[0] + (pole[0] - neutral[0]) * w,
			neutral[1] + (pole[1] - neutral[1]) * w,
			neutral[2] + (pole[2] - neutral[2]) * w
		);
		lut[3 * i] = r;
		lut[3 * i + 1] = g;
		lut[3 * i + 2] = b;
	}
	return { lut, iso: theme === 'light' ? 'rgba(20, 22, 28, 0.85)' : 'rgba(235, 238, 245, 0.85)' };
}

export function paintField(field: Pick<Field, 't'>, ramp: Ramp, img: ImageData): void {
	const d = img.data;
	const { t } = field;
	for (let k = 0; k < t.length; k++) {
		const idx = 3 * Math.max(0, Math.min(255, Math.round(t[k] * 255)));
		d[4 * k] = ramp.lut[idx];
		d[4 * k + 1] = ramp.lut[idx + 1];
		d[4 * k + 2] = ramp.lut[idx + 2];
		d[4 * k + 3] = 255;
	}
}

// ---------- isolines ----------

/**
 * Marching squares. Returns line segments [x0, y0, x1, y1, ...] in grid coordinates
 * (i ∈ [0, nx−1], j ∈ [0, ny−1]).
 */
export function isolines(
	field: Pick<Field, 'nx' | 'ny' | 't'>,
	levels: readonly number[]
): Float32Array {
	const { nx, ny, t } = field;
	const out: number[] = [];
	for (const lv of levels) {
		for (let j = 0; j < ny - 1; j++) {
			for (let i = 0; i < nx - 1; i++) {
				const a = t[j * nx + i]; // top-left
				const b = t[j * nx + i + 1]; // top-right
				const c = t[(j + 1) * nx + i + 1]; // bottom-right
				const d = t[(j + 1) * nx + i]; // bottom-left
				const code = (a > lv ? 8 : 0) | (b > lv ? 4 : 0) | (c > lv ? 2 : 0) | (d > lv ? 1 : 0);
				if (code === 0 || code === 15) continue;
				// edge crossing points
				const top = (): [number, number] => [i + (lv - a) / (b - a), j];
				const right = (): [number, number] => [i + 1, j + (lv - b) / (c - b)];
				const bottom = (): [number, number] => [i + (lv - d) / (c - d), j + 1];
				const left = (): [number, number] => [i, j + (lv - a) / (d - a)];
				const seg = (p: [number, number], q: [number, number]) => out.push(p[0], p[1], q[0], q[1]);
				switch (code) {
					case 1:
					case 14:
						seg(left(), bottom());
						break;
					case 2:
					case 13:
						seg(bottom(), right());
						break;
					case 3:
					case 12:
						seg(left(), right());
						break;
					case 4:
					case 11:
						seg(top(), right());
						break;
					case 6:
					case 9:
						seg(top(), bottom());
						break;
					case 7:
					case 8:
						seg(left(), top());
						break;
					case 5:
						seg(left(), top());
						seg(bottom(), right());
						break;
					case 10:
						seg(top(), right());
						seg(left(), bottom());
						break;
				}
			}
		}
	}
	return new Float32Array(out);
}
