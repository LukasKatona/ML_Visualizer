/**
 * Decision-boundary visualizer state, URL codec and training session.
 */
import { Rng } from '../math/rng.ts';
import { isDatasetId, type DatasetId, type Points } from './datasets.ts';
import { Knn } from './knn.ts';
import { Mlp, type Activation } from './mlp.ts';
import { VEC_DEFAULTS, VecOptimizer, type VecOptimizerId } from './optim.ts';
import { DecisionTree } from './tree.ts';

export type ModelId = 'logreg' | 'mlp' | 'knn' | 'tree';

export const MODELS: Record<ModelId, { name: string; trainable: boolean }> = {
	logreg: { name: 'Logistic regression', trainable: true },
	mlp: { name: 'Neural network (MLP)', trainable: true },
	knn: { name: 'k-nearest neighbours', trainable: false },
	tree: { name: 'Decision tree', trainable: false }
};

export const MODEL_IDS = Object.keys(MODELS) as ModelId[];
const isModelId = (s: string): s is ModelId => Object.hasOwn(MODELS, s);
const isOptId = (s: string): s is VecOptimizerId => Object.hasOwn(VEC_DEFAULTS, s);

export type ExtraPoint = [number, number, 0 | 1];

export interface DbState {
	dataset: DatasetId;
	n: number;
	noise: number;
	seed: number;
	/** User-placed training points. */
	extra: ExtraPoint[];
	model: ModelId;
	/** Polynomial feature degree (logistic regression and MLP input). */
	degree: number;
	layers: number;
	width: number;
	activation: Activation;
	optimizer: VecOptimizerId;
	lr: number;
	l2: number;
	k: number;
	maxDepth: number;
	/** Epochs per animation frame. */
	speed: number;
	maxEpochs: number;
}

export const DB_LIMITS = {
	n: { min: 20, max: 600 },
	noise: { min: 0, max: 0.6 },
	degree: { min: 1, max: 5 },
	layers: { min: 1, max: 3 },
	width: { min: 2, max: 32 },
	lr: { min: 1e-4, max: 3 },
	l2: { min: 0, max: 0.1 },
	k: { min: 1, max: 75 },
	maxDepth: { min: 1, max: 16 },
	speed: { min: 1, max: 50 },
	maxEpochs: { min: 50, max: 20_000 },
	extra: 150
} as const;

export function defaultDbState(): DbState {
	return {
		dataset: 'moons',
		n: 200,
		noise: 0.15,
		seed: 1,
		extra: [],
		model: 'mlp',
		degree: 1,
		layers: 1,
		width: 8,
		activation: 'tanh',
		optimizer: 'adam',
		lr: VEC_DEFAULTS.adam.lr,
		l2: 0,
		k: 15,
		maxDepth: 5,
		speed: 2,
		maxEpochs: 2000
	};
}

const r2 = (v: number) => String(Math.round(v * 100) / 100);
const fmt = (v: number) => String(Number(v.toPrecision(4)));
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export function encodeDb(s: DbState): string {
	const q: [string, string][] = [
		['d', s.dataset],
		['n', String(s.n)],
		['nz', fmt(s.noise)],
		['seed', String(s.seed)],
		['m', s.model]
	];
	if (s.model === 'logreg' || s.model === 'mlp') {
		q.push(['deg', String(s.degree)]);
		if (s.model === 'mlp') {
			q.push(['L', String(s.layers)], ['w', String(s.width)], ['act', s.activation]);
		}
		q.push(
			['opt', s.optimizer],
			['lr', fmt(s.lr)],
			['l2', fmt(s.l2)],
			['v', String(s.speed)],
			['E', String(s.maxEpochs)]
		);
	} else if (s.model === 'knn') q.push(['k', String(s.k)]);
	else q.push(['dep', String(s.maxDepth)]);
	if (s.extra.length)
		q.push(['pts', s.extra.map(([x, y, c]) => `${r2(x)},${r2(y)},${c}`).join(';')]);
	return q.map(([k, v]) => `${k}=${v}`).join('&');
}

export function decodeDb(search: string): DbState {
	const q = new URLSearchParams(search.startsWith('?') ? search.slice(1) : search);
	const s = defaultDbState();
	const L = DB_LIMITS;
	const num = (k: string): number | null => {
		const raw = q.get(k);
		if (raw === null || raw.trim() === '') return null;
		const v = Number(raw);
		return Number.isFinite(v) ? v : null;
	};
	const int = (k: string, lim: { min: number; max: number }, cur: number) => {
		const v = num(k);
		return v === null ? cur : Math.round(clamp(v, lim.min, lim.max));
	};
	const real = (k: string, lim: { min: number; max: number }, cur: number) => {
		const v = num(k);
		return v === null ? cur : clamp(v, lim.min, lim.max);
	};

	const d = q.get('d') ?? '';
	if (isDatasetId(d)) s.dataset = d;
	s.n = int('n', L.n, s.n);
	s.noise = real('nz', L.noise, s.noise);
	const seed = num('seed');
	if (seed !== null) s.seed = Math.abs(Math.trunc(seed)) >>> 0;
	const m = q.get('m') ?? '';
	if (isModelId(m)) s.model = m;
	s.degree = int('deg', L.degree, s.degree);
	s.layers = int('L', L.layers, s.layers);
	s.width = int('w', L.width, s.width);
	const act = q.get('act');
	if (act === 'tanh' || act === 'relu') s.activation = act;
	const opt = q.get('opt') ?? '';
	if (isOptId(opt)) {
		s.optimizer = opt;
		s.lr = VEC_DEFAULTS[opt].lr;
	}
	s.lr = real('lr', L.lr, s.lr);
	s.l2 = real('l2', L.l2, s.l2);
	s.k = int('k', L.k, s.k);
	s.maxDepth = int('dep', L.maxDepth, s.maxDepth);
	s.speed = int('v', L.speed, s.speed);
	s.maxEpochs = int('E', L.maxEpochs, s.maxEpochs);
	const pts = q.get('pts');
	if (pts) {
		for (const chunk of pts.split(';').slice(0, L.extra)) {
			const [x, y, c] = chunk.split(',').map(Number);
			if (Number.isFinite(x) && Number.isFinite(y) && (c === 0 || c === 1))
				s.extra.push([clamp(x, -1.6, 1.6), clamp(y, -1.6, 1.6), c]);
		}
	}
	return s;
}

// ---------- training session ----------

export interface Predictor {
	predict(x: number, y: number): number;
}

export type SessionStatus = 'idle' | 'training' | 'done' | 'diverged';

/**
 * Wraps a model with its data. Trainable models record per-epoch metrics into preallocated
 * buffers; k-NN and trees are fitted in the constructor.
 */
export class Session {
	readonly trainable: boolean;
	readonly predictor: Predictor;
	readonly mlp: Mlp | null = null;
	private readonly opt: VecOptimizer | null = null;
	private readonly seed: number;
	readonly loss: Float64Array;
	readonly trainAcc: Float64Array;
	readonly testAcc: Float64Array;
	/** Number of recorded epochs (epoch 0 = initialisation). */
	count = 0;
	status: SessionStatus = 'idle';
	/** Extra info for non-trainable models (tree leaves / depth). */
	readonly info: string;

	constructor(
		readonly st: DbState,
		readonly train: Points,
		readonly test: Points
	) {
		this.seed = st.seed;
		const cap = st.maxEpochs + 1;
		this.loss = new Float64Array(cap);
		this.trainAcc = new Float64Array(cap);
		this.testAcc = new Float64Array(cap);
		this.trainable = MODELS[st.model].trainable;
		if (st.model === 'knn') {
			this.predictor = new Knn(train, st.k);
			this.info = `k = ${Math.min(st.k, train.n)}`;
		} else if (st.model === 'tree') {
			const t = new DecisionTree(train, st.maxDepth);
			this.predictor = t;
			this.info = `${t.leaves} leaves, depth ${t.depth}`;
		} else {
			const hidden = st.model === 'mlp' ? Array(st.layers).fill(st.width) : [];
			const mlp = new Mlp({ degree: st.degree, hidden, activation: st.activation });
			this.mlp = mlp;
			this.predictor = mlp;
			const h = { ...VEC_DEFAULTS[st.optimizer], lr: st.lr };
			this.opt = new VecOptimizer(st.optimizer, h, mlp.params.length);
			this.info = `${mlp.params.length} parameters`;
		}
		this.reset();
	}

	reset(): void {
		this.count = 0;
		if (this.mlp && this.opt) {
			this.mlp.init(new Rng(this.seed ^ 0x5bd1e995));
			this.opt.reset();
			this.status = 'idle';
			this.record();
		} else {
			this.status = 'done';
		}
	}

	/** Evaluate metrics at the current parameters and store them (also computes gradients). */
	private record(): boolean {
		const m = this.mlp!;
		const { loss, acc } = m.lossAndGrad(this.train, this.st.l2);
		if (!Number.isFinite(loss)) return false;
		const i = this.count++;
		this.loss[i] = loss;
		this.trainAcc[i] = acc;
		this.testAcc[i] = m.accuracy(this.test);
		return true;
	}

	get epoch(): number {
		return Math.max(0, this.count - 1);
	}

	/** Run up to k epochs of full-batch training. Returns true while training can continue. */
	advance(k: number): boolean {
		if (!this.mlp || !this.opt || this.status === 'done' || this.status === 'diverged')
			return false;
		this.status = 'training';
		for (let e = 0; e < k; e++) {
			if (this.count > this.st.maxEpochs) {
				this.status = 'done';
				return false;
			}
			// gradients from the last record() are for the current parameters
			this.opt.step(this.mlp.params, this.mlp.grads);
			if (!this.record()) {
				this.status = 'diverged';
				return false;
			}
		}
		return true;
	}

	/** Train/test accuracy at the current state. */
	accuracies(): { train: number; test: number } {
		if (this.mlp) {
			const i = this.count - 1;
			return { train: this.trainAcc[i], test: this.testAcc[i] };
		}
		const acc = (d: Points) => {
			let c = 0;
			for (let s = 0; s < d.n; s++) {
				const p = this.predictor.predict(d.x[2 * s], d.x[2 * s + 1]);
				if (p >= 0.5 === (d.y[s] === 1)) c++;
			}
			return d.n ? c / d.n : 0;
		};
		return { train: acc(this.train), test: acc(this.test) };
	}
}
