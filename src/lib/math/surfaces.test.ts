import { describe, expect, it } from 'vitest';
import { Rng } from './rng.ts';
import { SURFACES, SURFACE_IDS } from './surfaces.ts';

/** Central-difference gradient with a step scaled to |x|. */
function numericGrad(f: (x: number, y: number) => number, x: number, y: number): [number, number] {
	const hx = 1e-6 * Math.max(1, Math.abs(x));
	const hy = 1e-6 * Math.max(1, Math.abs(y));
	return [(f(x + hx, y) - f(x - hx, y)) / (2 * hx), (f(x, y + hy) - f(x, y - hy)) / (2 * hy)];
}

describe('surface gradients match central differences', () => {
	const rng = new Rng(12345);
	const g = new Float64Array(2);

	for (const id of SURFACE_IDS) {
		it(id, () => {
			const s = SURFACES[id];
			const { xMin, xMax, yMin, yMax } = s.domain;
			for (let i = 0; i < 200; i++) {
				const x = xMin + (xMax - xMin) * rng.uniform();
				const y = yMin + (yMax - yMin) * rng.uniform();
				s.grad(x, y, g);
				const [nx, ny] = numericGrad(s.f, x, y);
				const scale = Math.max(1, Math.hypot(nx, ny));
				expect(Math.abs(g[0] - nx) / scale).toBeLessThan(1e-6);
				expect(Math.abs(g[1] - ny) / scale).toBeLessThan(1e-6);
			}
		});
	}
});

describe('known minima', () => {
	const g = new Float64Array(2);
	for (const id of SURFACE_IDS) {
		const s = SURFACES[id];
		for (const [mx, my] of s.minima) {
			it(`${id} at (${mx}, ${my}) is stationary with f = f*`, () => {
				expect(s.f(mx, my)).toBeCloseTo(s.fRef, 4);
				s.grad(mx, my, g);
				expect(Math.hypot(g[0], g[1])).toBeLessThan(1e-3);
			});
		}
	}

	it('saddle fRef is the domain minimum', () => {
		const s = SURFACES.saddle;
		expect(s.f(0, s.domain.yMax)).toBe(s.fRef);
	});

	it('default starts lie inside the domain', () => {
		for (const id of SURFACE_IDS) {
			const s = SURFACES[id];
			const [x, y] = s.defaultStart;
			expect(x).toBeGreaterThanOrEqual(s.domain.xMin);
			expect(x).toBeLessThanOrEqual(s.domain.xMax);
			expect(y).toBeGreaterThanOrEqual(s.domain.yMin);
			expect(y).toBeLessThanOrEqual(s.domain.yMax);
		}
	});
});

describe('Rng', () => {
	it('is reproducible for a seed', () => {
		const a = new Rng(7);
		const b = new Rng(7);
		for (let i = 0; i < 100; i++) expect(a.normal()).toBe(b.normal());
	});

	it('normal() has roughly zero mean and unit variance', () => {
		const r = new Rng(99);
		const n = 100_000;
		let s = 0;
		let s2 = 0;
		for (let i = 0; i < n; i++) {
			const z = r.normal();
			s += z;
			s2 += z * z;
		}
		const mean = s / n;
		expect(Math.abs(mean)).toBeLessThan(0.02);
		expect(Math.abs(s2 / n - mean * mean - 1)).toBeLessThan(0.02);
	});
});
