/**
 * Decision-boundary visualizer state, URL codec and training session.
 */
import { Rng } from '../math/rng.ts';
import { DATASETS, DOMAIN, isDatasetId, type DatasetId, type Points } from './datasets.ts';
import { Knn } from './knn.ts';
import { Mlp, type Activation } from './mlp.ts';
import { VEC_DEFAULTS, VecOptimizer, type VecOptimizerId } from './optim.ts';
import { TreeGrower, type SplitSeg } from './tree.ts';

export type ModelId = 'logreg' | 'mlp' | 'knn' | 'tree';

export interface ModelMeta {
	name: string;
	/** Gradient-trained (has loss, optimizer, epochs). */
	trainable: boolean;
	/** Transport labels and the unit of one step. */
	labels: { play: string; step: string; reset: string; unit: string };
}

export const MODELS: Record<ModelId, ModelMeta> = {
	logreg: {
		name: 'Logistic regression',
		trainable: true,
		labels: { play: 'Train', step: 'Step', reset: 'Re-init', unit: 'Epoch' }
	},
	mlp: {
		name: 'Neural network (MLP)',
		trainable: true,
		labels: { play: 'Train', step: 'Step', reset: 'Re-init', unit: 'Epoch' }
	},
	knn: {
		name: 'k-nearest neighbours',
		trainable: false,
		labels: { play: 'Replay', step: 'Store point', reset: 'Empty', unit: 'Stored points' }
	},
	tree: {
		name: 'Decision tree',
		trainable: false,
		labels: { play: 'Grow', step: 'Split', reset: 'Root only', unit: 'Splits' }
	}
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
		noise: DATASETS.moons.noise,
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
		q.push(['opt', s.optimizer], ['lr', fmt(s.lr)], ['l2', fmt(s.l2)], ['E', String(s.maxEpochs)]);
	} else if (s.model === 'knn') q.push(['k', String(s.k)]);
	else q.push(['dep', String(s.maxDepth)]);
	q.push(['v', String(s.speed)]);
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
	if (isDatasetId(d)) {
		s.dataset = d;
		s.noise = DATASETS[d].noise;
	}
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

/** Optional visual hooks a predictor may provide to the canvas. */
export interface SplitSource {
	splits: SplitSeg[];
}
export interface NeighbourSource {
	kEff: number;
	data: Points;
	neighbors(x: number, y: number, out: Int32Array): number;
}

export const hasSplits = (p: Predictor): p is Predictor & SplitSource => 'splits' in p;
export const hasNeighbours = (p: Predictor): p is Predictor & NeighbourSource => 'neighbors' in p;

export type SessionStatus = 'idle' | 'training' | 'done' | 'diverged';

/** Non-gradient models built step by step: k-NN stores one point per step, trees add one split. */
interface Stepper {
	reset(): void;
	step(): boolean;
	finish(): void;
}

class KnnStepper implements Stepper {
	constructor(readonly knn: Knn) {}
	reset() {
		this.knn.size = 0;
	}
	step() {
		if (this.knn.size >= this.knn.data.n) return false;
		this.knn.size++;
		return true;
	}
	finish() {
		this.knn.size = this.knn.data.n;
	}
}

/**
 * Wraps a model with its data and records per-step metrics into preallocated buffers.
 * Gradient-trained models start at initialisation (step 0). Steppers (k-NN, tree) start fully
 * fitted with no history; reset() rewinds them to step 0 so they can be replayed.
 */
export class Session {
	readonly trainable: boolean;
	readonly predictor: Predictor;
	readonly mlp: Mlp | null = null;
	private readonly opt: VecOptimizer | null = null;
	private readonly stepper: Stepper | null = null;
	private readonly knn: Knn | null = null;
	private readonly tree: TreeGrower | null = null;
	private readonly seed: number;
	readonly loss: Float64Array;
	readonly trainAcc: Float64Array;
	readonly testAcc: Float64Array;
	/** Number of recorded steps (index 0 = initial state). */
	count = 0;
	status: SessionStatus = 'idle';

	constructor(
		readonly st: DbState,
		readonly train: Points,
		readonly test: Points
	) {
		this.seed = st.seed;
		this.trainable = MODELS[st.model].trainable;
		// capacity: epochs for gradient models; at most n stored points / n − 1 splits otherwise
		const cap = (this.trainable ? st.maxEpochs : train.n) + 1;
		this.loss = new Float64Array(this.trainable ? cap : 0);
		this.trainAcc = new Float64Array(cap);
		this.testAcc = new Float64Array(cap);
		if (st.model === 'knn') {
			const knn = new Knn(train, st.k);
			this.knn = knn;
			this.predictor = knn;
			this.stepper = new KnnStepper(knn);
		} else if (st.model === 'tree') {
			const D = DOMAIN;
			const t = new TreeGrower(train, st.maxDepth, 1, {
				x0: D.xMin,
				x1: D.xMax,
				y0: D.yMin,
				y1: D.yMax
			});
			this.tree = t;
			this.predictor = t;
			this.stepper = t;
		} else {
			const hidden = st.model === 'mlp' ? Array(st.layers).fill(st.width) : [];
			const mlp = new Mlp({ degree: st.degree, hidden, activation: st.activation });
			this.mlp = mlp;
			this.predictor = mlp;
			const h = { ...VEC_DEFAULTS[st.optimizer], lr: st.lr };
			this.opt = new VecOptimizer(st.optimizer, h, mlp.params.length);
		}
		if (this.stepper) {
			this.stepper.finish();
			this.status = 'done';
		} else this.reset();
	}

	/** Number of training points k-NN has stored (null for other models). */
	get stored(): number | null {
		return this.knn ? this.knn.size : null;
	}

	/** Short description of the model's current size. */
	get info(): string {
		if (this.knn) return `${this.knn.size}/${this.train.n} stored · k = ${this.knn.kEff}`;
		if (this.tree) return `${this.tree.leaves} leaves · depth ${this.tree.depth}`;
		return `${this.mlp!.params.length} parameters`;
	}

	reset(): void {
		this.count = 0;
		if (this.mlp && this.opt) {
			this.mlp.init(new Rng(this.seed ^ 0x5bd1e995));
			this.opt.reset();
		} else this.stepper!.reset();
		this.status = 'idle';
		this.record();
	}

	/** Evaluate metrics at the current state and store them. */
	private record(): boolean {
		const i = this.count;
		if (i >= this.trainAcc.length) return false;
		if (this.mlp) {
			// also leaves ∇ in mlp.grads for the next optimizer step
			const { loss, acc } = this.mlp.lossAndGrad(this.train, this.st.l2);
			if (!Number.isFinite(loss)) return false;
			this.loss[i] = loss;
			this.trainAcc[i] = acc;
			this.testAcc[i] = this.mlp.accuracy(this.test);
		} else {
			this.trainAcc[i] = this.accuracy(this.train);
			this.testAcc[i] = this.accuracy(this.test);
		}
		this.count++;
		return true;
	}

	private accuracy(d: Points): number {
		let c = 0;
		for (let s = 0; s < d.n; s++) {
			const p = this.predictor.predict(d.x[2 * s], d.x[2 * s + 1]);
			if (p >= 0.5 === (d.y[s] === 1)) c++;
		}
		return d.n ? c / d.n : 0;
	}

	/** Steps taken since reset (0 for a fitted stepper that has not been replayed). */
	get step(): number {
		return Math.max(0, this.count - 1);
	}

	/** Advance up to k steps (epochs, stored points or splits). Returns true while more remain. */
	advance(k: number): boolean {
		if (this.status === 'done' || this.status === 'diverged') return false;
		this.status = 'training';
		for (let e = 0; e < k; e++) {
			if (this.mlp && this.opt) {
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
			} else {
				if (!this.stepper!.step() || !this.record()) {
					this.status = 'done';
					return false;
				}
			}
		}
		return true;
	}

	/** Train/test accuracy at the current state. */
	accuracies(): { train: number; test: number } {
		if (this.count > 0) {
			const i = this.count - 1;
			return { train: this.trainAcc[i], test: this.testAcc[i] };
		}
		return { train: this.accuracy(this.train), test: this.accuracy(this.test) };
	}
}
