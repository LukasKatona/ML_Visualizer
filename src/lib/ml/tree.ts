/**
 * CART decision tree for binary labels: greedy axis-aligned splits minimising weighted Gini
 * impurity, threshold at the midpoint between consecutive distinct feature values.
 * Leaves predict the class-1 fraction of their training points.
 */
import type { Points } from './datasets.ts';

interface Leaf {
	leaf: true;
	p: number;
	n: number;
}
interface Split {
	leaf: false;
	feat: 0 | 1;
	thr: number;
	left: Node;
	right: Node;
	n: number;
	p: number;
}
type Node = Leaf | Split;

const gini = (ones: number, n: number) => {
	if (n === 0) return 0;
	const p = ones / n;
	return 2 * p * (1 - p);
};

export class DecisionTree {
	readonly root: Node;
	leaves = 0;
	depth = 0;

	constructor(
		private readonly data: Points,
		readonly maxDepth: number,
		readonly minLeaf = 1
	) {
		const idx = Array.from({ length: data.n }, (_, i) => i);
		this.root = this.build(idx, 0);
	}

	private build(idx: number[], depth: number): Node {
		const { x, y } = this.data;
		let ones = 0;
		for (const i of idx) ones += y[i];
		const n = idx.length;
		const p = n ? ones / n : 0.5;
		if (depth > this.depth) this.depth = depth;
		if (depth >= this.maxDepth || n < 2 * this.minLeaf || ones === 0 || ones === n) {
			this.leaves++;
			return { leaf: true, p, n };
		}
		const parent = gini(ones, n);
		let best = { gain: 1e-12, feat: 0 as 0 | 1, thr: 0 };
		for (const feat of [0, 1] as const) {
			const sorted = idx.slice().sort((a, b) => x[2 * a + feat] - x[2 * b + feat]);
			let leftOnes = 0;
			for (let k = 0; k < n - 1; k++) {
				leftOnes += y[sorted[k]];
				const nl = k + 1;
				const nr = n - nl;
				if (nl < this.minLeaf || nr < this.minLeaf) continue;
				const v = x[2 * sorted[k] + feat];
				const vNext = x[2 * sorted[k + 1] + feat];
				if (vNext === v) continue;
				const w = (nl * gini(leftOnes, nl) + nr * gini(ones - leftOnes, nr)) / n;
				const gain = parent - w;
				if (gain > best.gain) best = { gain, feat, thr: 0.5 * (v + vNext) };
			}
		}
		if (best.gain <= 1e-12) {
			this.leaves++;
			return { leaf: true, p, n };
		}
		const L: number[] = [];
		const R: number[] = [];
		for (const i of idx) (x[2 * i + best.feat] <= best.thr ? L : R).push(i);
		return {
			leaf: false,
			feat: best.feat,
			thr: best.thr,
			n,
			p,
			left: this.build(L, depth + 1),
			right: this.build(R, depth + 1)
		};
	}

	predict(x: number, y: number): number {
		let node = this.root;
		while (!node.leaf) node = (node.feat === 0 ? x : y) <= node.thr ? node.left : node.right;
		return node.p;
	}
}
