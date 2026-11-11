/**
 * k-nearest-neighbour classifier: P(y = 1 | x) = fraction of class-1 points among the k nearest.
 * Only the first `size` training points are "stored", so the model can be built step by step
 * (k-NN's training is memorisation).
 */
import type { Points } from './datasets.ts';

export class Knn {
	private readonly dist: Float64Array;
	private readonly near: Int32Array;
	/** Number of stored training points (≤ data.n). */
	size: number;

	constructor(
		readonly data: Points,
		readonly k: number
	) {
		this.k = Math.max(1, Math.min(k, data.n));
		this.size = data.n;
		this.dist = new Float64Array(this.k);
		this.near = new Int32Array(this.k);
	}

	/** Effective k given the stored points. */
	get kEff(): number {
		return Math.min(this.k, this.size);
	}

	/** Fills `out` with the indices of the k nearest stored points (nearest first); returns count. */
	neighbors(x: number, y: number, out: Int32Array = this.near): number {
		const { data, dist } = this;
		const k = this.kEff;
		let filled = 0;
		for (let s = 0; s < this.size; s++) {
			const dx = data.x[2 * s] - x;
			const dy = data.x[2 * s + 1] - y;
			const d = dx * dx + dy * dy;
			if (filled < k || d < dist[k - 1]) {
				let i = filled < k ? filled++ : k - 1;
				while (i > 0 && dist[i - 1] > d) {
					dist[i] = dist[i - 1];
					out[i] = out[i - 1];
					i--;
				}
				dist[i] = d;
				out[i] = s;
			}
		}
		return filled;
	}

	predict(x: number, y: number): number {
		const m = this.neighbors(x, y);
		if (m === 0) return 0.5;
		let ones = 0;
		for (let i = 0; i < m; i++) ones += this.data.y[this.near[i]];
		return ones / m;
	}
}
