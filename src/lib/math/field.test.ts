import { describe, expect, it } from 'vitest';
import { isolines, makeRamp, normalise, sampleField } from './field.ts';
import { SURFACES } from './surfaces.ts';

describe('field', () => {
	it('normalises log-loss into [0, 1] with the minimum at 0', () => {
		const f = sampleField(SURFACES.quadratic, 61, 31);
		let lo = Infinity;
		let hi = -Infinity;
		for (const v of f.t) {
			lo = Math.min(lo, v);
			hi = Math.max(hi, v);
		}
		expect(lo).toBeCloseTo(0, 6);
		expect(hi).toBeCloseTo(1, 6);
		expect(normalise(f, 0)).toBeCloseTo(0, 6);
	});

	it('isoline crossings lie on the requested level set', () => {
		const s = SURFACES.noisy; // anisotropic quadratic; check level crossings lie on the ellipse
		const f = sampleField(s, 201, 121);
		const level = normalise(f, 2);
		const segs = isolines(f, [level]);
		expect(segs.length).toBeGreaterThan(40);
		const { xMin, xMax, yMin, yMax } = s.domain;
		for (let k = 0; k < segs.length; k += 2) {
			const x = xMin + ((xMax - xMin) * segs[k]) / (f.nx - 1);
			const y = yMax - ((yMax - yMin) * segs[k + 1]) / (f.ny - 1);
			expect(s.f(x, y)).toBeCloseTo(2, 1);
		}
	});

	it('ramps are monotone in lightness', () => {
		for (const theme of ['light', 'dark'] as const) {
			const { lut } = makeRamp(theme);
			const lum = (i: number) => lut[3 * i] + lut[3 * i + 1] + lut[3 * i + 2];
			const dir = theme === 'light' ? -1 : 1;
			for (let i = 8; i < 256; i += 8)
				expect(dir * (lum(i) - lum(i - 8))).toBeGreaterThanOrEqual(0);
		}
	});
});
