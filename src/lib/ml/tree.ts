/**
 * CART decision tree for binary labels: axis-aligned splits minimising weighted Gini impurity,
 * threshold at the midpoint between consecutive distinct feature values. Leaves predict the
 * class-1 fraction of their training points.
 *
 * `TreeGrower` builds the tree one split at a time, best-first: each step expands the frontier
 * leaf with the largest weighted impurity decrease n·ΔG. Because every split is a local decision,
 * growing to completion yields exactly the same tree as recursive depth-first CART.
 */
import type { Points } from './datasets.ts';

export interface Bounds {
	x0: number;
	x1: number;
	y0: number;
	y1: number;
}

/** A split drawn as a segment: feat = 0 → vertical line x = thr over [lo, hi] in y. */
export interface SplitSeg {
	feat: 0 | 1;
	thr: number;
	lo: number;
	hi: number;
	depth: number;
}

interface BestSplit {
	score: number; // n · ΔG
	feat: 0 | 1;
	thr: number;
}

interface Node {
	leaf: boolean;
	p: number;
	n: number;
	depth: number;
	bounds: Bounds;
	idx: number[] | null;
	best: BestSplit | null;
	feat: 0 | 1;
	thr: number;
	left: Node | null;
	right: Node | null;
}

const gini = (ones: number, n: number) => {
	if (n === 0) return 0;
	const p = ones / n;
	return 2 * p * (1 - p);
};

const BIG: Bounds = { x0: -1e9, x1: 1e9, y0: -1e9, y1: 1e9 };

export class TreeGrower {
	root!: Node;
	splits: SplitSeg[] = [];
	leaves = 1;
	depth = 0;
	private frontier: Node[] = [];

	constructor(
		private readonly data: Points,
		readonly maxDepth: number,
		readonly minLeaf = 1,
		private readonly view: Bounds = BIG
	) {
		this.reset();
	}

	reset(): void {
		const idx = Array.from({ length: this.data.n }, (_, i) => i);
		this.root = this.makeLeaf(idx, 0, { ...this.view });
		this.splits = [];
		this.leaves = 1;
		this.depth = 0;
		this.frontier = this.root.best ? [this.root] : [];
	}

	private makeLeaf(idx: number[], depth: number, bounds: Bounds): Node {
		const { y } = this.data;
		let ones = 0;
		for (const i of idx) ones += y[i];
		const n = idx.length;
		const node: Node = {
			leaf: true,
			p: n ? ones / n : 0.5,
			n,
			depth,
			bounds,
			idx,
			best: null,
			feat: 0,
			thr: 0,
			left: null,
			right: null
		};
		if (depth < this.maxDepth && n >= 2 * this.minLeaf && ones > 0 && ones < n) {
			node.best = this.findSplit(idx, ones);
		}
		return node;
	}

	private findSplit(idx: number[], ones: number): BestSplit | null {
		const { x, y } = this.data;
		const n = idx.length;
		const parent = gini(ones, n);
		let best: BestSplit | null = null;
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
				const score = n * (parent - w);
				if (score > 1e-12 && (!best || score > best.score))
					best = { score, feat, thr: 0.5 * (v + vNext) };
			}
		}
		return best;
	}

	/** Expand the most valuable frontier leaf. Returns false when the tree is complete. */
	step(): boolean {
		if (!this.frontier.length) return false;
		let bi = 0;
		for (let i = 1; i < this.frontier.length; i++)
			if (this.frontier[i].best!.score > this.frontier[bi].best!.score) bi = i;
		const node = this.frontier[bi];
		this.frontier.splice(bi, 1);
		const { feat, thr } = node.best!;
		const { x } = this.data;
		const L: number[] = [];
		const R: number[] = [];
		for (const i of node.idx!) (x[2 * i + feat] <= thr ? L : R).push(i);
		const b = node.bounds;
		const lb = feat === 0 ? { ...b, x1: thr } : { ...b, y1: thr };
		const rb = feat === 0 ? { ...b, x0: thr } : { ...b, y0: thr };
		node.leaf = false;
		node.feat = feat;
		node.thr = thr;
		node.idx = null;
		node.left = this.makeLeaf(L, node.depth + 1, lb);
		node.right = this.makeLeaf(R, node.depth + 1, rb);
		for (const c of [node.left, node.right]) if (c.best) this.frontier.push(c);
		this.splits.push(
			feat === 0
				? { feat, thr, lo: b.y0, hi: b.y1, depth: node.depth }
				: { feat, thr, lo: b.x0, hi: b.x1, depth: node.depth }
		);
		this.leaves++;
		this.depth = Math.max(this.depth, node.depth + 1);
		return true;
	}

	finish(): void {
		while (this.step());
	}

	get done(): boolean {
		return this.frontier.length === 0;
	}

	predict(x: number, y: number): number {
		let node = this.root;
		while (!node.leaf) node = (node.feat === 0 ? x : y) <= node.thr ? node.left! : node.right!;
		return node.p;
	}
}

/** A fully grown tree (convenience wrapper). */
export class DecisionTree extends TreeGrower {
	constructor(data: Points, maxDepth: number, minLeaf = 1, view?: Bounds) {
		super(data, maxDepth, minLeaf, view);
		this.finish();
	}
}
