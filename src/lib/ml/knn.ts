/** k-nearest-neighbour classifier: P(y = 1 | x) = fraction of class-1 points among the k nearest. */
import type { Points } from './datasets.ts';

export class Knn {
	private readonly dist: Float64Array;
	private readonly lab: Uint8Array;

	constructor(
		private readonly data: Points,
		readonly k: number
	) {
		this.k = Math.max(1, Math.min(k, data.n));
		this.dist = new Float64Array(this.k);
		this.lab = new Uint8Array(this.k);
	}

	predict(x: number, y: number): number {
		const { data, k, dist, lab } = this;
		if (data.n === 0) return 0.5;
		let filled = 0;
		for (let s = 0; s < data.n; s++) {
			const dx = data.x[2 * s] - x;
			const dy = data.x[2 * s + 1] - y;
			const d = dx * dx + dy * dy;
			if (filled < k) {
				// insertion into sorted prefix
				let i = filled++;
				while (i > 0 && dist[i - 1] > d) {
					dist[i] = dist[i - 1];
					lab[i] = lab[i - 1];
					i--;
				}
				dist[i] = d;
				lab[i] = data.y[s];
			} else if (d < dist[k - 1]) {
				let i = k - 1;
				while (i > 0 && dist[i - 1] > d) {
					dist[i] = dist[i - 1];
					lab[i] = lab[i - 1];
					i--;
				}
				dist[i] = d;
				lab[i] = data.y[s];
			}
		}
		let ones = 0;
		for (let i = 0; i < filled; i++) ones += lab[i];
		return ones / filled;
	}
}
