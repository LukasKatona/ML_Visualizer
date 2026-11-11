/**
 * Optimizer update rules for a 2-parameter problem.
 *
 * Each optimizer mutates `p` (parameters, length 2) and `s` (state, length STATE_SIZE) in place.
 * `grad(x, y, out)` evaluates the (possibly noisy) gradient; `g` is a scratch buffer for it.
 * `t` is the 1-based step index. No allocations happen inside `step`.
 */

export type OptimizerId =
	'sgd' | 'momentum' | 'nesterov' | 'adagrad' | 'rmsprop' | 'adam' | 'adamw';

export type HyperKey = 'lr' | 'beta' | 'beta1' | 'beta2' | 'eps' | 'wd';

export type Hyper = Partial<Record<HyperKey, number>> & { lr: number };

export type GradFn = (x: number, y: number, out: Float64Array) => void;

export interface Optimizer {
	id: OptimizerId;
	name: string;
	/** Which family supplies the default learning rate for a surface. */
	family: 'sgd' | 'adaptive';
	/** Hyperparameters this optimizer uses, in display order. */
	hypers: readonly HyperKey[];
	/** Defaults for everything except lr (which is surface-dependent). */
	defaults: Omit<Hyper, 'lr'>;
	step(p: Float64Array, s: Float64Array, g: Float64Array, t: number, h: Hyper, grad: GradFn): void;
}

/** Per-run state slots: [a0, a1, b0, b1] — first and second moment (or velocity) per coordinate. */
export const STATE_SIZE = 4;

const sgd: Optimizer = {
	id: 'sgd',
	name: 'SGD',
	family: 'sgd',
	hypers: ['lr'],
	defaults: {},
	step(p, _s, g, _t, h, grad) {
		grad(p[0], p[1], g);
		p[0] -= h.lr * g[0];
		p[1] -= h.lr * g[1];
	}
};

const momentum: Optimizer = {
	id: 'momentum',
	name: 'Momentum',
	family: 'sgd',
	hypers: ['lr', 'beta'],
	defaults: { beta: 0.9 },
	step(p, s, g, _t, h, grad) {
		const b = h.beta!;
		grad(p[0], p[1], g);
		s[0] = b * s[0] + g[0];
		s[1] = b * s[1] + g[1];
		p[0] -= h.lr * s[0];
		p[1] -= h.lr * s[1];
	}
};

const nesterov: Optimizer = {
	id: 'nesterov',
	name: 'Nesterov',
	family: 'sgd',
	hypers: ['lr', 'beta'],
	defaults: { beta: 0.9 },
	step(p, s, g, _t, h, grad) {
		const b = h.beta!;
		// gradient at the look-ahead point θ − η β v
		grad(p[0] - h.lr * b * s[0], p[1] - h.lr * b * s[1], g);
		s[0] = b * s[0] + g[0];
		s[1] = b * s[1] + g[1];
		p[0] -= h.lr * s[0];
		p[1] -= h.lr * s[1];
	}
};

const adagrad: Optimizer = {
	id: 'adagrad',
	name: 'AdaGrad',
	family: 'adaptive',
	hypers: ['lr', 'eps'],
	defaults: { eps: 1e-8 },
	step(p, s, g, _t, h, grad) {
		const eps = h.eps!;
		grad(p[0], p[1], g);
		s[2] += g[0] * g[0];
		s[3] += g[1] * g[1];
		p[0] -= (h.lr * g[0]) / (Math.sqrt(s[2]) + eps);
		p[1] -= (h.lr * g[1]) / (Math.sqrt(s[3]) + eps);
	}
};

const rmsprop: Optimizer = {
	id: 'rmsprop',
	name: 'RMSProp',
	family: 'adaptive',
	hypers: ['lr', 'beta', 'eps'],
	defaults: { beta: 0.9, eps: 1e-8 },
	step(p, s, g, _t, h, grad) {
		const b = h.beta!;
		const eps = h.eps!;
		grad(p[0], p[1], g);
		s[2] = b * s[2] + (1 - b) * g[0] * g[0];
		s[3] = b * s[3] + (1 - b) * g[1] * g[1];
		p[0] -= (h.lr * g[0]) / (Math.sqrt(s[2]) + eps);
		p[1] -= (h.lr * g[1]) / (Math.sqrt(s[3]) + eps);
	}
};

function adamCore(
	p: Float64Array,
	s: Float64Array,
	g: Float64Array,
	t: number,
	h: Hyper,
	grad: GradFn,
	wd: number
): void {
	const b1 = h.beta1!;
	const b2 = h.beta2!;
	const eps = h.eps!;
	grad(p[0], p[1], g);
	s[0] = b1 * s[0] + (1 - b1) * g[0];
	s[1] = b1 * s[1] + (1 - b1) * g[1];
	s[2] = b2 * s[2] + (1 - b2) * g[0] * g[0];
	s[3] = b2 * s[3] + (1 - b2) * g[1] * g[1];
	const c1 = 1 - Math.pow(b1, t);
	const c2 = 1 - Math.pow(b2, t);
	const u0 = s[0] / c1 / (Math.sqrt(s[2] / c2) + eps);
	const u1 = s[1] / c1 / (Math.sqrt(s[3] / c2) + eps);
	p[0] -= h.lr * (u0 + wd * p[0]);
	p[1] -= h.lr * (u1 + wd * p[1]);
}

const adam: Optimizer = {
	id: 'adam',
	name: 'Adam',
	family: 'adaptive',
	hypers: ['lr', 'beta1', 'beta2', 'eps'],
	defaults: { beta1: 0.9, beta2: 0.999, eps: 1e-8 },
	step(p, s, g, t, h, grad) {
		adamCore(p, s, g, t, h, grad, 0);
	}
};

const adamw: Optimizer = {
	id: 'adamw',
	name: 'AdamW',
	family: 'adaptive',
	hypers: ['lr', 'beta1', 'beta2', 'eps', 'wd'],
	defaults: { beta1: 0.9, beta2: 0.999, eps: 1e-8, wd: 0.01 },
	step(p, s, g, t, h, grad) {
		adamCore(p, s, g, t, h, grad, h.wd!);
	}
};

export const OPTIMIZERS: Record<OptimizerId, Optimizer> = {
	sgd,
	momentum,
	nesterov,
	adagrad,
	rmsprop,
	adam,
	adamw
};

export const OPTIMIZER_IDS = Object.keys(OPTIMIZERS) as OptimizerId[];

export function isOptimizerId(s: string): s is OptimizerId {
	return Object.hasOwn(OPTIMIZERS, s);
}

export const HYPER_META: Record<
	HyperKey,
	{ label: string; latex: string; min: number; max: number; log: boolean }
> = {
	lr: { label: 'Learning rate', latex: String.raw`\eta`, min: 1e-5, max: 1, log: true },
	beta: { label: 'β', latex: String.raw`\beta`, min: 0, max: 0.999, log: false },
	beta1: { label: 'β₁', latex: String.raw`\beta_1`, min: 0, max: 0.999, log: false },
	beta2: { label: 'β₂', latex: String.raw`\beta_2`, min: 0.8, max: 0.9999, log: false },
	eps: { label: 'ε', latex: String.raw`\epsilon`, min: 1e-10, max: 1, log: true },
	wd: { label: 'Weight decay λ', latex: String.raw`\lambda`, min: 1e-4, max: 1, log: true }
};

/** Full hyperparameter set for an optimizer on a surface with the given default lrs. */
export function defaultHyper(id: OptimizerId, lr: { sgd: number; adaptive: number }): Hyper {
	const o = OPTIMIZERS[id];
	return { ...o.defaults, lr: lr[o.family] };
}
