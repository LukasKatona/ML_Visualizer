/** Vector versions of SGD / Momentum / Adam for training models with many parameters. */

export type VecOptimizerId = 'sgd' | 'momentum' | 'adam';

export interface VecHyper {
	lr: number;
	beta: number;
	beta1: number;
	beta2: number;
	eps: number;
}

export const VEC_DEFAULTS: Record<VecOptimizerId, VecHyper> = {
	sgd: { lr: 0.5, beta: 0.9, beta1: 0.9, beta2: 0.999, eps: 1e-8 },
	momentum: { lr: 0.1, beta: 0.9, beta1: 0.9, beta2: 0.999, eps: 1e-8 },
	adam: { lr: 0.03, beta: 0.9, beta1: 0.9, beta2: 0.999, eps: 1e-8 }
};

export class VecOptimizer {
	private readonly m: Float64Array;
	private readonly v: Float64Array;
	private t = 0;

	constructor(
		readonly kind: VecOptimizerId,
		readonly h: VecHyper,
		size: number
	) {
		this.m = new Float64Array(size);
		this.v = new Float64Array(size);
	}

	reset(): void {
		this.m.fill(0);
		this.v.fill(0);
		this.t = 0;
	}

	step(p: Float64Array, g: Float64Array): void {
		const { lr } = this.h;
		const n = p.length;
		this.t++;
		if (this.kind === 'sgd') {
			for (let i = 0; i < n; i++) p[i] -= lr * g[i];
		} else if (this.kind === 'momentum') {
			const b = this.h.beta;
			const m = this.m;
			for (let i = 0; i < n; i++) {
				m[i] = b * m[i] + g[i];
				p[i] -= lr * m[i];
			}
		} else {
			const { beta1: b1, beta2: b2, eps } = this.h;
			const c1 = 1 - Math.pow(b1, this.t);
			const c2 = 1 - Math.pow(b2, this.t);
			const m = this.m;
			const v = this.v;
			for (let i = 0; i < n; i++) {
				m[i] = b1 * m[i] + (1 - b1) * g[i];
				v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
				p[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + eps);
			}
		}
	}
}
