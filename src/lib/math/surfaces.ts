/**
 * 2D loss surfaces with analytic gradients.
 *
 * `grad` writes into a caller-provided buffer so the simulation hot loop never allocates.
 */

export type SurfaceId = 'quadratic' | 'rosenbrock' | 'himmelblau' | 'beale' | 'saddle' | 'noisy';

export interface Domain {
	xMin: number;
	xMax: number;
	yMin: number;
	yMax: number;
}

export interface Surface {
	id: SurfaceId;
	name: string;
	/** KaTeX source for f(x, y). */
	latex: string;
	/** One-line technical note shown in the UI. */
	note: string;
	f(x: number, y: number): number;
	/** Writes ∇f(x, y) into out[0], out[1]. */
	grad(x: number, y: number, out: Float64Array): void;
	domain: Domain;
	/** Known global minimisers (for markers). Empty if unbounded below. */
	minima: ReadonlyArray<readonly [number, number]>;
	/**
	 * Reference value f* for the loss chart (plotted as f − f*). The global minimum where it
	 * exists; for surfaces unbounded below, the minimum over the displayed domain.
	 */
	fRef: number;
	defaultStart: readonly [number, number];
	/** Default learning rates: plain-gradient family (SGD/Momentum/Nesterov) and adaptive family. */
	lr: { sgd: number; adaptive: number };
	/** Default gradient noise σ (only the stochastic surface uses noise). */
	noise: number;
}

/** Condition number of the ill-conditioned quadratic. */
export const QUAD_KAPPA = 25;

const quadratic: Surface = {
	id: 'quadratic',
	name: 'Ill-conditioned quadratic',
	latex: String.raw`f(x,y)=\tfrac12\left(x^2+${QUAD_KAPPA}\,y^2\right)`,
	note: `Hessian diag(1, ${QUAD_KAPPA}), κ = ${QUAD_KAPPA}. GD is stable for η < 2/${QUAD_KAPPA}; the slow direction converges at rate 1 − η.`,
	f: (x, y) => 0.5 * (x * x + QUAD_KAPPA * y * y),
	grad: (x, y, out) => {
		out[0] = x;
		out[1] = QUAD_KAPPA * y;
	},
	domain: { xMin: -6, xMax: 6, yMin: -3, yMax: 3 },
	minima: [[0, 0]],
	fRef: 0,
	defaultStart: [-5, 2],
	lr: { sgd: 0.04, adaptive: 0.1 },
	noise: 0
};

const rosenbrock: Surface = {
	id: 'rosenbrock',
	name: 'Rosenbrock',
	latex: String.raw`f(x,y)=(1-x)^2+100\,(y-x^2)^2`,
	note: 'Narrow curved valley; finding it is easy, travelling along it is slow. Minimum at (1, 1).',
	f: (x, y) => {
		const a = 1 - x;
		const b = y - x * x;
		return a * a + 100 * b * b;
	},
	grad: (x, y, out) => {
		const b = y - x * x;
		out[0] = -2 * (1 - x) - 400 * x * b;
		out[1] = 200 * b;
	},
	domain: { xMin: -2, xMax: 2, yMin: -1, yMax: 3 },
	minima: [[1, 1]],
	fRef: 0,
	defaultStart: [-1.5, 2.5],
	lr: { sgd: 4e-4, adaptive: 0.05 },
	noise: 0
};

const himmelblau: Surface = {
	id: 'himmelblau',
	name: 'Himmelblau',
	latex: String.raw`f(x,y)=(x^2+y-11)^2+(x+y^2-7)^2`,
	note: 'Four global minima (f = 0) and a local maximum near (−0.27, −0.92): which basin you land in depends on the start and the optimizer’s dynamics.',
	f: (x, y) => {
		const a = x * x + y - 11;
		const b = x + y * y - 7;
		return a * a + b * b;
	},
	grad: (x, y, out) => {
		const a = x * x + y - 11;
		const b = x + y * y - 7;
		out[0] = 4 * x * a + 2 * b;
		out[1] = 2 * a + 4 * y * b;
	},
	domain: { xMin: -5, xMax: 5, yMin: -5, yMax: 5 },
	minima: [
		[3, 2],
		[-2.805118, 3.131312],
		[-3.77931, -3.283186],
		[3.584428, -1.848126]
	],
	fRef: 0,
	defaultStart: [-0.3, -0.9],
	lr: { sgd: 0.01, adaptive: 0.1 },
	noise: 0
};

const beale: Surface = {
	id: 'beale',
	name: 'Beale',
	latex: String.raw`f(x,y)=(1.5-x+xy)^2+(2.25-x+xy^2)^2+(2.625-x+xy^3)^2`,
	note: 'Flat plateaus next to steep walls (gradients span ~6 orders of magnitude). Minimum at (3, 0.5).',
	f: (x, y) => {
		const y2 = y * y;
		const a = 1.5 - x + x * y;
		const b = 2.25 - x + x * y2;
		const c = 2.625 - x + x * y2 * y;
		return a * a + b * b + c * c;
	},
	grad: (x, y, out) => {
		const y2 = y * y;
		const y3 = y2 * y;
		const a = 1.5 - x + x * y;
		const b = 2.25 - x + x * y2;
		const c = 2.625 - x + x * y3;
		out[0] = 2 * (a * (y - 1) + b * (y2 - 1) + c * (y3 - 1));
		out[1] = 2 * x * (a + 2 * b * y + 3 * c * y2);
	},
	domain: { xMin: -4.5, xMax: 4.5, yMin: -4.5, yMax: 4.5 },
	minima: [[3, 0.5]],
	fRef: 0,
	defaultStart: [0.7, 1.4],
	lr: { sgd: 1e-3, adaptive: 0.05 },
	noise: 0
};

const saddle: Surface = {
	id: 'saddle',
	name: 'Saddle point',
	latex: String.raw`f(x,y)=x^2-y^2`,
	note: 'Unbounded below; ∇f = 0 at the origin. Starting near y = 0, the escape speed depends on how fast the optimizer amplifies the tiny y-gradient.',
	f: (x, y) => x * x - y * y,
	grad: (x, y, out) => {
		out[0] = 2 * x;
		out[1] = -2 * y;
	},
	domain: { xMin: -2, xMax: 2, yMin: -2, yMax: 2 },
	minima: [],
	fRef: -4,
	defaultStart: [-1.9, 1e-3],
	lr: { sgd: 0.05, adaptive: 0.02 },
	noise: 0
};

const NOISY_A = 1;
const NOISY_B = 6;
const noisy: Surface = {
	id: 'noisy',
	name: 'Stochastic gradients',
	latex: String.raw`f(x,y)=\tfrac12\left(x^2+${NOISY_B}\,y^2\right),\quad \tilde g=\nabla f+\sigma\,\xi,\ \xi\sim\mathcal N(0,I)`,
	note: 'True loss is a quadratic; every step sees a noisy gradient (a minibatch proxy). Constant-η methods settle into a noise ball whose radius scales with η·σ.',
	f: (x, y) => 0.5 * (NOISY_A * x * x + NOISY_B * y * y),
	grad: (x, y, out) => {
		out[0] = NOISY_A * x;
		out[1] = NOISY_B * y;
	},
	domain: { xMin: -5, xMax: 5, yMin: -3, yMax: 3 },
	minima: [[0, 0]],
	fRef: 0,
	defaultStart: [-4.2, 2.2],
	lr: { sgd: 0.05, adaptive: 0.05 },
	noise: 2
};

export const SURFACES: Record<SurfaceId, Surface> = {
	quadratic,
	rosenbrock,
	himmelblau,
	beale,
	saddle,
	noisy
};

export const SURFACE_IDS = Object.keys(SURFACES) as SurfaceId[];

export function isSurfaceId(s: string): s is SurfaceId {
	return Object.hasOwn(SURFACES, s);
}
