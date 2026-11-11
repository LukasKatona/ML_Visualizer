import { describe, expect, it } from 'vitest';
import { SURFACES } from '../math/surfaces.ts';
import {
	OPTIMIZERS,
	OPTIMIZER_IDS,
	STATE_SIZE,
	defaultHyper,
	type GradFn,
	type Hyper,
	type OptimizerId
} from './optimizers.ts';

/** Gradient of f = ½(a x² + b y²). */
const quadGrad =
	(a: number, b: number): GradFn =>
	(x, y, out) => {
		out[0] = a * x;
		out[1] = b * y;
	};

function stepOnce(id: OptimizerId, p0: [number, number], h: Hyper, grad: GradFn, steps = 1) {
	const p = new Float64Array(p0);
	const s = new Float64Array(STATE_SIZE);
	const g = new Float64Array(2);
	for (let t = 1; t <= steps; t++) OPTIMIZERS[id].step(p, s, g, t, h, grad);
	return { p: [p[0], p[1]], s: Array.from(s) };
}

describe('single-step update math (hand-computed)', () => {
	// f = ½(x² + 4y²) at (1, 1): ∇f = (1, 4)
	const grad = quadGrad(1, 4);

	it('SGD: θ ← θ − η g', () => {
		const { p } = stepOnce('sgd', [1, 1], { lr: 0.1 }, grad);
		expect(p[0]).toBeCloseTo(0.9, 12);
		expect(p[1]).toBeCloseTo(0.6, 12);
	});

	it('Momentum: two steps', () => {
		// t=1: v=(1,4), θ=(0.9,0.6). t=2: g=(0.9,2.4), v=(0.5+0.9, 2+2.4)=(1.4,4.4), θ=(0.76,0.16)
		const { p, s } = stepOnce('momentum', [1, 1], { lr: 0.1, beta: 0.5 }, grad, 2);
		expect(s[0]).toBeCloseTo(1.4, 12);
		expect(s[1]).toBeCloseTo(4.4, 12);
		expect(p[0]).toBeCloseTo(0.76, 12);
		expect(p[1]).toBeCloseTo(0.16, 12);
	});

	it('Nesterov: gradient is taken at the look-ahead point', () => {
		// t=1: v=0 so lookahead=θ → same as momentum: v=(1,4), θ=(0.9,0.6)
		// t=2: lookahead = θ − η β v = (0.9−0.05, 0.6−0.2) = (0.85, 0.4); g = (0.85, 1.6)
		//      v = (0.5+0.85, 2+1.6) = (1.35, 3.6); θ = (0.9−0.135, 0.6−0.36) = (0.765, 0.24)
		const { p } = stepOnce('nesterov', [1, 1], { lr: 0.1, beta: 0.5 }, grad, 2);
		expect(p[0]).toBeCloseTo(0.765, 12);
		expect(p[1]).toBeCloseTo(0.24, 12);
	});

	it('AdaGrad: first step moves each coordinate by ≈ η', () => {
		const { p, s } = stepOnce('adagrad', [1, 1], { lr: 0.1, eps: 0 }, grad);
		expect(s[2]).toBe(1);
		expect(s[3]).toBe(16);
		expect(p[0]).toBeCloseTo(0.9, 12);
		expect(p[1]).toBeCloseTo(0.9, 12);
	});

	it('RMSProp: v = β v + (1−β) g²', () => {
		// v = (0.1·1, 0.1·16) = (0.1, 1.6); step = η g/√v = 0.1·(1/√0.1, 4/√1.6)
		const { p } = stepOnce('rmsprop', [1, 1], { lr: 0.1, beta: 0.9, eps: 0 }, grad);
		expect(p[0]).toBeCloseTo(1 - 0.1 / Math.sqrt(0.1), 12);
		expect(p[1]).toBeCloseTo(1 - 0.4 / Math.sqrt(1.6), 12);
	});

	it('Adam: bias correction makes the first step exactly η·sign(g)', () => {
		const { p } = stepOnce('adam', [1, 1], { lr: 0.1, beta1: 0.9, beta2: 0.999, eps: 0 }, grad);
		expect(p[0]).toBeCloseTo(0.9, 12);
		expect(p[1]).toBeCloseTo(0.9, 12);
	});

	it('Adam: second step matches the formula', () => {
		const b1 = 0.9,
			b2 = 0.999,
			lr = 0.1;
		const { p } = stepOnce('adam', [1, 1], { lr, beta1: b1, beta2: b2, eps: 0 }, grad, 2);
		// after step 1: θ = (0.9, 0.9); g2 = (0.9, 3.6)
		const g1 = [1, 4];
		const g2 = [0.9, 3.6];
		for (let i = 0; i < 2; i++) {
			const m = b1 * (1 - b1) * g1[i] + (1 - b1) * g2[i];
			const v = b2 * (1 - b2) * g1[i] ** 2 + (1 - b2) * g2[i] ** 2;
			const mh = m / (1 - b1 ** 2);
			const vh = v / (1 - b2 ** 2);
			expect(p[i]).toBeCloseTo(0.9 - (lr * mh) / Math.sqrt(vh), 12);
		}
	});

	it('AdamW: decoupled weight decay adds η λ θ', () => {
		const h = { lr: 0.1, beta1: 0.9, beta2: 0.999, eps: 0, wd: 0.5 };
		const { p } = stepOnce('adamw', [1, 1], h, grad);
		expect(p[0]).toBeCloseTo(1 - 0.1 * (1 + 0.5), 12);
		expect(p[1]).toBeCloseTo(1 - 0.1 * (1 + 0.5), 12);
	});
});

describe('convergence on the ill-conditioned quadratic', () => {
	const surf = SURFACES.quadratic;
	const steps: Record<OptimizerId, number> = {
		sgd: 600,
		momentum: 600,
		nesterov: 600,
		adagrad: 3000,
		rmsprop: 3000,
		adam: 3000,
		adamw: 3000
	};

	for (const id of OPTIMIZER_IDS) {
		it(`${id} reaches the minimum from the default start`, () => {
			const h = defaultHyper(id, surf.lr);
			if (id === 'adamw') h.wd = 0.01; // decay toward 0 coincides with the minimum here
			// RMSProp/Adam with constant η oscillate at amplitude ~η near the optimum; shrink η
			// to test convergence rather than the size of the limit cycle.
			if (id === 'rmsprop' || id === 'adam' || id === 'adamw') h.lr = 0.01;
			const { p } = stepOnce(id, [...surf.defaultStart], h, surf.grad, steps[id]);
			expect(Math.hypot(p[0], p[1])).toBeLessThan(0.05);
			expect(surf.f(p[0], p[1])).toBeLessThan(1e-3);
		});
	}

	it('plain GD diverges above the stability limit η > 2/L', () => {
		const L = 25;
		const { p } = stepOnce('sgd', [1, 1], { lr: 2.05 / L }, surf.grad, 200);
		expect(Math.abs(p[1])).toBeGreaterThan(1);
	});

	it('momentum stability bounds: heavy ball ηL < 2(1+β), Nesterov ηL < 2(1+β)/(1+2β)', () => {
		const L = 25;
		const beta = 0.9;
		const hb = (2 * (1 + beta)) / L;
		const nag = (2 * (1 + beta)) / (1 + 2 * beta) / L;
		const run = (id: OptimizerId, lr: number) =>
			Math.abs(stepOnce(id, [0, 1], { lr, beta }, surf.grad, 2000).p[1]);
		expect(run('momentum', 0.97 * hb)).toBeLessThan(1);
		expect(run('momentum', 1.03 * hb)).toBeGreaterThan(1);
		expect(run('nesterov', 0.97 * nag)).toBeLessThan(1);
		expect(run('nesterov', 1.03 * nag)).toBeGreaterThan(1);
	});

	it('GD rate on the quadratic is |1 − η λᵢ| per coordinate', () => {
		const lr = 0.05;
		const { p } = stepOnce('sgd', [1, 1], { lr }, surf.grad, 10);
		expect(p[0]).toBeCloseTo((1 - lr) ** 10, 12);
		expect(p[1]).toBeCloseTo((1 - 25 * lr) ** 10, 12);
	});
});
