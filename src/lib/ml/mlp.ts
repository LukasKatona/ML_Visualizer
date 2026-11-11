/**
 * Fully connected binary classifier on 2D inputs:
 *   φ(x) = polynomial features of total degree ≤ d  (d = 1 is the identity)
 *   h_0 = φ(x),  h_l = act(W_l h_{l−1} + b_l),  p = σ(w_Lᵀ h_{L−1} + b_L)
 * With no hidden layers this is exactly (polynomial) logistic regression.
 *
 * Loss: mean binary cross-entropy + (λ/2)·Σ‖W‖² (weights only, not biases).
 * All buffers are preallocated; forward/backward do not allocate.
 */
import type { Rng } from '../math/rng.ts';
import type { Points } from './datasets.ts';

export type Activation = 'tanh' | 'relu';

export interface MlpConfig {
	/** Polynomial feature degree (1–5). */
	degree: number;
	/** Hidden layer widths; empty for logistic regression. */
	hidden: number[];
	activation: Activation;
}

/** Number of polynomial features of total degree 1..d in two variables. */
export function featureCount(d: number): number {
	return ((d + 1) * (d + 2)) / 2 - 1;
}

/** Writes [x, y, x², xy, y², x³, ...] (all monomials of degree 1..d) into out. */
export function polyFeatures(x: number, y: number, d: number, out: Float64Array): void {
	let k = 0;
	for (let deg = 1; deg <= d; deg++) {
		for (let i = deg; i >= 0; i--) out[k++] = x ** i * y ** (deg - i);
	}
}

const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
/** log(1 + e^z), numerically stable. */
const softplus = (z: number) => (z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z)));

export class Mlp {
	readonly sizes: number[];
	readonly params: Float64Array;
	readonly grads: Float64Array;
	/** Offsets of W_l and b_l in `params` for each layer l = 1..L (index l−1). */
	private readonly wOff: number[] = [];
	private readonly bOff: number[] = [];
	/** Post-activation outputs per layer (index 0 = features). */
	private readonly acts: Float64Array[];
	/** Backprop deltas per layer (index l for layer l). */
	private readonly deltas: Float64Array[];

	constructor(readonly cfg: MlpConfig) {
		this.sizes = [featureCount(cfg.degree), ...cfg.hidden, 1];
		let off = 0;
		for (let l = 1; l < this.sizes.length; l++) {
			this.wOff.push(off);
			off += this.sizes[l] * this.sizes[l - 1];
			this.bOff.push(off);
			off += this.sizes[l];
		}
		this.params = new Float64Array(off);
		this.grads = new Float64Array(off);
		this.acts = this.sizes.map((s) => new Float64Array(s));
		this.deltas = this.sizes.map((s) => new Float64Array(s));
	}

	get layers(): number {
		return this.sizes.length - 1;
	}

	/** Xavier (tanh / output) or He (ReLU) normal init; zero biases. */
	init(rng: Rng): void {
		this.params.fill(0);
		for (let l = 1; l < this.sizes.length; l++) {
			const fanIn = this.sizes[l - 1];
			const isOut = l === this.sizes.length - 1;
			const gain = !isOut && this.cfg.activation === 'relu' ? 2 : 1;
			const std = Math.sqrt(gain / fanIn);
			const w = this.wOff[l - 1];
			for (let k = 0; k < this.sizes[l] * fanIn; k++) this.params[w + k] = std * rng.normal();
		}
	}

	/** Output logit z for input (x, y); fills the activation buffers. */
	private forwardLogit(x: number, y: number): number {
		const P = this.params;
		polyFeatures(x, y, this.cfg.degree, this.acts[0]);
		const L = this.layers;
		const relu = this.cfg.activation === 'relu';
		let z = 0;
		for (let l = 1; l <= L; l++) {
			const inA = this.acts[l - 1];
			const out = this.acts[l];
			const nIn = this.sizes[l - 1];
			const w = this.wOff[l - 1];
			const b = this.bOff[l - 1];
			for (let j = 0; j < this.sizes[l]; j++) {
				let s = P[b + j];
				const row = w + j * nIn;
				for (let i = 0; i < nIn; i++) s += P[row + i] * inA[i];
				if (l === L) {
					z = s;
					out[j] = s;
				} else out[j] = relu ? (s > 0 ? s : 0) : Math.tanh(s);
			}
		}
		return z;
	}

	/** P(y = 1 | x). */
	predict(x: number, y: number): number {
		return sigmoid(this.forwardLogit(x, y));
	}

	/**
	 * Mean BCE + L2 over `data`; writes ∇ into `grads`. Returns loss and training accuracy.
	 */
	lossAndGrad(data: Points, l2: number): { loss: number; acc: number } {
		const P = this.params;
		const G = this.grads;
		G.fill(0);
		const L = this.layers;
		const relu = this.cfg.activation === 'relu';
		const inv = 1 / Math.max(1, data.n);
		let loss = 0;
		let correct = 0;
		for (let s = 0; s < data.n; s++) {
			const t = data.y[s];
			const z = this.forwardLogit(data.x[2 * s], data.x[2 * s + 1]);
			loss += softplus(z) - t * z;
			const p = sigmoid(z);
			if (p >= 0.5 === (t === 1)) correct++;
			// output delta: dℓ/dz = p − t (scaled by 1/n for the mean)
			this.deltas[L][0] = (p - t) * inv;
			for (let l = L; l >= 1; l--) {
				const nIn = this.sizes[l - 1];
				const nOut = this.sizes[l];
				const w = this.wOff[l - 1];
				const b = this.bOff[l - 1];
				const d = this.deltas[l];
				const a = this.acts[l - 1];
				for (let j = 0; j < nOut; j++) {
					const dj = d[j];
					G[b + j] += dj;
					const row = w + j * nIn;
					for (let i = 0; i < nIn; i++) G[row + i] += dj * a[i];
				}
				if (l > 1) {
					const dPrev = this.deltas[l - 1];
					for (let i = 0; i < nIn; i++) {
						let s2 = 0;
						for (let j = 0; j < nOut; j++) s2 += P[w + j * nIn + i] * d[j];
						const ai = a[i];
						dPrev[i] = s2 * (relu ? (ai > 0 ? 1 : 0) : 1 - ai * ai);
					}
				}
			}
		}
		loss *= inv;
		if (l2 > 0) {
			for (let l = 1; l <= L; l++) {
				const w = this.wOff[l - 1];
				const end = w + this.sizes[l] * this.sizes[l - 1];
				for (let k = w; k < end; k++) {
					loss += 0.5 * l2 * P[k] * P[k];
					G[k] += l2 * P[k];
				}
			}
		}
		return { loss, acc: correct * inv };
	}

	accuracy(data: Points): number {
		let c = 0;
		for (let s = 0; s < data.n; s++) {
			const p = this.predict(data.x[2 * s], data.x[2 * s + 1]);
			if (p >= 0.5 === (data.y[s] === 1)) c++;
		}
		return data.n ? c / data.n : 0;
	}
}
