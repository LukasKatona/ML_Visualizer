/**
 * Seeded 2D binary-classification datasets, roughly normalised to [-1, 1]².
 * Points are stored interleaved (x0, y0, x1, y1, ...) with labels in {0, 1}.
 */
import { Rng } from '../math/rng.ts';

export type DatasetId = 'blobs' | 'xor' | 'moons' | 'circles' | 'spirals';

export interface Points {
	/** Interleaved coordinates, length 2n. */
	x: Float64Array;
	/** Labels in {0, 1}, length n. */
	y: Uint8Array;
	n: number;
}

export interface Split {
	train: Points;
	test: Points;
}

export interface DatasetMeta {
	id: DatasetId;
	name: string;
	note: string;
	/**
	 * Default Gaussian jitter σ. Per dataset, because the class separation differs a lot
	 * (adjacent spiral arms are only ~0.35 apart, the blobs ~1.1).
	 */
	noise: number;
}

export const DATASETS: Record<DatasetId, DatasetMeta> = {
	blobs: {
		id: 'blobs',
		name: 'Gaussian blobs',
		note: 'Two isotropic Gaussians: (nearly) linearly separable, so the Bayes boundary is a line.',
		noise: 0.1
	},
	xor: {
		id: 'xor',
		name: 'XOR',
		note: 'Label = sign(x)·sign(y). No linear model can separate it; a product feature xy or one hidden layer can.',
		noise: 0.1
	},
	moons: {
		id: 'moons',
		name: 'Two moons',
		note: 'Interleaving half-circles: a smooth non-linear boundary that is easy to overfit at high noise.',
		noise: 0.12
	},
	circles: {
		id: 'circles',
		name: 'Concentric circles',
		note: 'Radially separable: degree-2 features (x² + y²) suffice; trees need many axis-aligned splits.',
		noise: 0.1
	},
	spirals: {
		id: 'spirals',
		name: 'Spirals',
		note: 'Two interleaved spirals: a hard boundary that needs depth/width and many epochs.',
		noise: 0.04
	}
};

export const DATASET_IDS = Object.keys(DATASETS) as DatasetId[];

export function isDatasetId(s: string): s is DatasetId {
	return Object.hasOwn(DATASETS, s);
}

/** Display / sampling domain shared by all datasets. */
export const DOMAIN = { xMin: -1.6, xMax: 1.6, yMin: -1.6, yMax: 1.6 } as const;

/** Fraction of generated points held out for testing. */
export const TEST_FRACTION = 0.3;

function sample(id: DatasetId, cls: 0 | 1, rng: Rng, out: [number, number]): void {
	switch (id) {
		case 'blobs': {
			const c = cls ? 0.45 : -0.45;
			out[0] = c + 0.3 * rng.normal();
			out[1] = c * 0.8 + 0.3 * rng.normal();
			return;
		}
		case 'xor': {
			// rejection-free: pick a quadrant consistent with the label
			const sx = rng.uniform() < 0.5 ? -1 : 1;
			const sy = cls ? sx : -sx;
			out[0] = sx * (0.08 + 0.92 * rng.uniform());
			out[1] = sy * (0.08 + 0.92 * rng.uniform());
			return;
		}
		case 'moons': {
			const t = Math.PI * rng.uniform();
			const [mx, my] = cls ? [1 - Math.cos(t), 0.5 - Math.sin(t)] : [Math.cos(t), Math.sin(t)];
			out[0] = (mx - 0.5) * 0.85;
			out[1] = (my - 0.25) * 0.85;
			return;
		}
		case 'circles': {
			const a = 2 * Math.PI * rng.uniform();
			const r = cls ? 0.95 : 0.45;
			out[0] = r * Math.cos(a);
			out[1] = r * Math.sin(a);
			return;
		}
		case 'spirals': {
			const u = rng.uniform();
			const t = 0.25 + 2.6 * Math.PI * u;
			const r = 0.08 + 0.92 * u;
			const a = t + (cls ? Math.PI : 0);
			out[0] = r * Math.cos(a);
			out[1] = r * Math.sin(a);
			return;
		}
	}
}

/**
 * Generate `n` points (balanced classes, shuffled) with Gaussian jitter of std `noise`,
 * then split off TEST_FRACTION as a test set.
 */
export function generate(id: DatasetId, n: number, noise: number, seed: number): Split {
	const rng = new Rng(seed);
	const xs = new Float64Array(2 * n);
	const ys = new Uint8Array(n);
	const p: [number, number] = [0, 0];
	for (let i = 0; i < n; i++) {
		const cls = (i % 2) as 0 | 1;
		sample(id, cls, rng, p);
		xs[2 * i] = p[0] + noise * rng.normal();
		xs[2 * i + 1] = p[1] + noise * rng.normal();
		ys[i] = cls;
	}
	// Fisher–Yates shuffle (keeps the split unbiased)
	for (let i = n - 1; i > 0; i--) {
		const j = Math.floor(rng.uniform() * (i + 1));
		const tx = xs[2 * i];
		const ty = xs[2 * i + 1];
		xs[2 * i] = xs[2 * j];
		xs[2 * i + 1] = xs[2 * j + 1];
		xs[2 * j] = tx;
		xs[2 * j + 1] = ty;
		const tl = ys[i];
		ys[i] = ys[j];
		ys[j] = tl;
	}
	const nTest = Math.round(n * TEST_FRACTION);
	const nTrain = n - nTest;
	return {
		train: { x: xs.slice(0, 2 * nTrain), y: ys.slice(0, nTrain), n: nTrain },
		test: { x: xs.slice(2 * nTrain), y: ys.slice(nTrain), n: nTest }
	};
}

/** Append extra (user-placed) points to a point set. */
export function withExtra(
	base: Points,
	extra: readonly (readonly [number, number, 0 | 1])[]
): Points {
	if (!extra.length) return base;
	const n = base.n + extra.length;
	const x = new Float64Array(2 * n);
	const y = new Uint8Array(n);
	x.set(base.x);
	y.set(base.y);
	extra.forEach(([px, py, c], k) => {
		x[2 * (base.n + k)] = px;
		x[2 * (base.n + k) + 1] = py;
		y[base.n + k] = c;
	});
	return { x, y, n };
}
