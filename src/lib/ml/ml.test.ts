import { describe, expect, it } from 'vitest';
import { Rng } from '../math/rng.ts';
import {
	DATASETS,
	DATASET_IDS,
	TEST_FRACTION,
	generate,
	withExtra,
	type Points
} from './datasets.ts';
import { Knn } from './knn.ts';
import { Mlp, featureCount, polyFeatures, type MlpConfig } from './mlp.ts';
import { VEC_DEFAULTS, VecOptimizer, type VecOptimizerId } from './optim.ts';
import { DecisionTree, TreeGrower } from './tree.ts';

function trainAcc(predict: (x: number, y: number) => number, d: Points) {
	let c = 0;
	for (let i = 0; i < d.n; i++)
		if (predict(d.x[2 * i], d.x[2 * i + 1]) >= 0.5 === (d.y[i] === 1)) c++;
	return c / d.n;
}

function train(cfg: MlpConfig, d: Points, epochs: number, opt: VecOptimizerId = 'adam', l2 = 0) {
	const m = new Mlp(cfg);
	m.init(new Rng(3));
	const o = new VecOptimizer(opt, { ...VEC_DEFAULTS[opt] }, m.params.length);
	for (let e = 0; e < epochs; e++) {
		m.lossAndGrad(d, l2);
		o.step(m.params, m.grads);
	}
	return m;
}

describe('datasets', () => {
	it('are deterministic per seed, balanced and split by TEST_FRACTION', () => {
		for (const id of DATASET_IDS) {
			const a = generate(id, 200, 0.1, 7);
			const b = generate(id, 200, 0.1, 7);
			expect(a.train.x).toEqual(b.train.x);
			expect(a.test.y).toEqual(b.test.y);
			expect(a.test.n).toBe(Math.round(200 * TEST_FRACTION));
			expect(a.train.n + a.test.n).toBe(200);
			const ones = [...a.train.y, ...a.test.y].reduce((s, v) => s + v, 0);
			expect(ones).toBe(100);
		}
	});

	it('every dataset is clearly learnable at its default noise (k-NN, k = 5)', () => {
		for (const id of DATASET_IDS) {
			const { train, test } = generate(id, 400, DATASETS[id].noise, 1);
			const knn = new Knn(train, 5);
			expect(
				trainAcc((x, y) => knn.predict(x, y), test),
				id
			).toBeGreaterThan(0.9);
		}
	});

	it('XOR labels match sign(x)·sign(y) without noise', () => {
		const { train } = generate('xor', 300, 0, 1);
		for (let i = 0; i < train.n; i++) {
			const pos = train.x[2 * i] * train.x[2 * i + 1] > 0;
			expect(train.y[i]).toBe(pos ? 1 : 0);
		}
	});

	it('withExtra appends user points', () => {
		const { train } = generate('blobs', 20, 0, 1);
		const e = withExtra(train, [[0.1, 0.2, 1]]);
		expect(e.n).toBe(train.n + 1);
		expect(e.x[2 * train.n + 1]).toBe(0.2);
		expect(e.y[train.n]).toBe(1);
	});
});

describe('MLP', () => {
	it('polynomial features: count and ordering', () => {
		expect(featureCount(1)).toBe(2);
		expect(featureCount(3)).toBe(9);
		const out = new Float64Array(featureCount(2));
		polyFeatures(2, 3, 2, out);
		expect(Array.from(out)).toEqual([2, 3, 4, 6, 9]);
	});

	for (const cfg of [
		{ degree: 1, hidden: [], activation: 'tanh' },
		{ degree: 2, hidden: [5, 4], activation: 'tanh' },
		{ degree: 1, hidden: [6], activation: 'relu' }
	] as MlpConfig[]) {
		it(`backprop matches central differences (${JSON.stringify(cfg)})`, () => {
			const { train: d } = generate('moons', 40, 0.2, 5);
			const m = new Mlp(cfg);
			m.init(new Rng(11));
			const l2 = 0.01;
			m.lossAndGrad(d, l2);
			const g = m.grads.slice();
			const h = 1e-6;
			for (let k = 0; k < m.params.length; k++) {
				const v = m.params[k];
				m.params[k] = v + h;
				const lp = m.lossAndGrad(d, l2).loss;
				m.params[k] = v - h;
				const lm = m.lossAndGrad(d, l2).loss;
				m.params[k] = v;
				const num = (lp - lm) / (2 * h);
				expect(Math.abs(g[k] - num)).toBeLessThan(1e-6 * Math.max(1, Math.abs(num)));
			}
		});
	}

	it('logistic regression separates blobs', () => {
		const { train: d } = generate('blobs', 200, 0, 2);
		const m = train({ degree: 1, hidden: [], activation: 'tanh' }, d, 300);
		expect(m.accuracy(d)).toBeGreaterThan(0.95);
	});

	it('linear logistic regression cannot solve XOR; degree 2 can', () => {
		const { train: d } = generate('xor', 300, 0, 4);
		const lin = train({ degree: 1, hidden: [], activation: 'tanh' }, d, 500);
		expect(lin.accuracy(d)).toBeLessThan(0.7);
		const quad = train({ degree: 2, hidden: [], activation: 'tanh' }, d, 500);
		expect(quad.accuracy(d)).toBeGreaterThan(0.95);
	});

	it('a one-hidden-layer MLP solves XOR and moons', () => {
		for (const id of ['xor', 'moons'] as const) {
			const { train: d } = generate(id, 300, 0.05, 6);
			const m = train({ degree: 1, hidden: [8], activation: 'tanh' }, d, 800);
			expect(m.accuracy(d), id).toBeGreaterThan(0.95);
		}
	});

	it('loss decreases under plain SGD and momentum', () => {
		const { train: d } = generate('moons', 200, 0.1, 8);
		for (const opt of ['sgd', 'momentum'] as const) {
			const m = new Mlp({ degree: 1, hidden: [8], activation: 'tanh' });
			m.init(new Rng(1));
			const l0 = m.lossAndGrad(d, 0).loss;
			const o = new VecOptimizer(opt, { ...VEC_DEFAULTS[opt] }, m.params.length);
			for (let e = 0; e < 300; e++) {
				m.lossAndGrad(d, 0);
				o.step(m.params, m.grads);
			}
			expect(m.lossAndGrad(d, 0).loss, opt).toBeLessThan(0.6 * l0);
		}
	});
});

describe('k-NN', () => {
	it('k = 1 memorises the training set', () => {
		const { train: d } = generate('spirals', 200, 0.1, 3);
		const knn = new Knn(d, 1);
		expect(trainAcc((x, y) => knn.predict(x, y), d)).toBe(1);
	});

	it('stores points incrementally and reports neighbours nearest-first', () => {
		const d: Points = {
			x: new Float64Array([0, 0, 1, 0, 2, 0, 3, 0]),
			y: new Uint8Array([1, 1, 0, 0]),
			n: 4
		};
		const knn = new Knn(d, 3);
		knn.size = 0;
		expect(knn.predict(0, 0)).toBe(0.5);
		knn.size = 2;
		expect(knn.kEff).toBe(2);
		expect(knn.predict(3, 0)).toBe(1); // only the two class-1 points are stored
		knn.size = 4;
		const out = new Int32Array(3);
		expect(knn.neighbors(2.2, 0, out)).toBe(3);
		expect(Array.from(out)).toEqual([2, 3, 1]);
	});

	it('returns the class-1 fraction among the k nearest', () => {
		const d: Points = {
			x: new Float64Array([0, 0, 1, 0, 2, 0, 3, 0]),
			y: new Uint8Array([1, 1, 0, 0]),
			n: 4
		};
		expect(new Knn(d, 3).predict(0.1, 0)).toBeCloseTo(2 / 3, 12);
		expect(new Knn(d, 2).predict(2.9, 0)).toBe(0);
		expect(new Knn(d, 10).predict(0, 0)).toBe(0.5); // k clamped to n
	});
});

describe('decision tree', () => {
	it('unlimited depth fits the training set exactly', () => {
		const { train: d } = generate('moons', 200, 0.25, 9);
		const t = new DecisionTree(d, 50);
		expect(trainAcc((x, y) => t.predict(x, y), d)).toBe(1);
	});

	it('depth 1 is a stump with two leaves', () => {
		const { train: d } = generate('blobs', 200, 0.1, 9);
		const t = new DecisionTree(d, 1);
		expect(t.leaves).toBe(2);
		expect(t.depth).toBe(1);
		expect(trainAcc((x, y) => t.predict(x, y), d)).toBeGreaterThan(0.8);
	});

	it('best-first growth reaches the same tree as recursive CART, one split per step', () => {
		const { train: d, test } = generate('circles', 300, 0.1, 4);
		const full = new DecisionTree(d, 6);
		const g = new TreeGrower(d, 6);
		expect(g.leaves).toBe(1);
		let steps = 0;
		while (g.step()) steps++;
		expect(steps).toBe(full.leaves - 1);
		expect(g.splits).toHaveLength(steps);
		for (let i = 0; i < test.n; i++) {
			const x = test.x[2 * i];
			const y = test.x[2 * i + 1];
			expect(g.predict(x, y)).toBe(full.predict(x, y));
		}
		g.reset();
		expect(g.leaves).toBe(1);
		expect(g.predict(0, 0)).toBeCloseTo(0.5, 1);
	});

	it('a depth-1 tree is complete after exactly one split', () => {
		const { train: d } = generate('blobs', 200, 0.1, 2);
		const g = new TreeGrower(d, 1);
		g.step();
		expect(g.splits).toHaveLength(1);
		expect(g.step()).toBe(false);
	});

	it('finds the obvious axis-aligned split', () => {
		const d: Points = {
			x: new Float64Array([-1, 0, -0.5, 1, 0.5, -1, 1, 0.3]),
			y: new Uint8Array([0, 0, 1, 1]),
			n: 4
		};
		const t = new DecisionTree(d, 1);
		expect(t.root.leaf).toBe(false);
		if (!t.root.leaf) {
			expect(t.root.feat).toBe(0);
			expect(t.root.thr).toBe(0);
		}
	});
});
