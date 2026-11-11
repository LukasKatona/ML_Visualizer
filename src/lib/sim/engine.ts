/**
 * Optimizer race simulation. Framework-free; all buffers are preallocated at construction so
 * `advance()` (called from requestAnimationFrame) performs no allocations.
 */
import { Rng, deriveSeed } from '../math/rng.ts';
import type { Surface } from '../math/surfaces.ts';
import {
	OPTIMIZERS,
	STATE_SIZE,
	type GradFn,
	type Hyper,
	type Optimizer,
	type OptimizerId
} from '../optimizers/optimizers.ts';

export type RunStatus = 'running' | 'converged' | 'diverged' | 'escaped' | 'done';

export interface RunConfig {
	id: OptimizerId;
	h: Hyper;
}

/** Gradient-norm threshold below which a noise-free run counts as converged. */
export const CONVERGED_GRAD_NORM = 1e-7;

export class Run {
	readonly id: OptimizerId;
	readonly opt: Optimizer;
	readonly h: Hyper;
	readonly p = new Float64Array(2);
	readonly s = new Float64Array(STATE_SIZE);
	readonly g = new Float64Array(2);
	/** Interleaved trajectory [x0, y0, x1, y1, ...], `count` points valid. */
	readonly traj: Float64Array;
	/** Loss gap f(θ_t) − f* for each recorded point. */
	readonly loss: Float64Array;
	count = 0;
	status: RunStatus = 'running';
	private readonly rng: Rng;
	private readonly seed: number;
	readonly grad: GradFn;

	constructor(
		cfg: RunConfig,
		private readonly surface: Surface,
		maxSteps: number,
		noise: number,
		seed: number
	) {
		this.id = cfg.id;
		this.opt = OPTIMIZERS[cfg.id];
		this.h = { ...cfg.h };
		this.traj = new Float64Array(2 * (maxSteps + 1));
		this.loss = new Float64Array(maxSteps + 1);
		this.seed = seed;
		this.rng = new Rng(seed);
		const sg = surface.grad;
		const rng = this.rng;
		this.grad =
			noise > 0
				? (x, y, out) => {
						sg(x, y, out);
						out[0] += noise * rng.normal();
						out[1] += noise * rng.normal();
					}
				: sg;
	}

	reset(x: number, y: number): void {
		this.p[0] = x;
		this.p[1] = y;
		this.s.fill(0);
		this.rng.reseed(this.seed);
		this.count = 0;
		this.status = 'running';
		this.record();
	}

	record(): void {
		const i = this.count++;
		this.traj[2 * i] = this.p[0];
		this.traj[2 * i + 1] = this.p[1];
		this.loss[i] = this.surface.f(this.p[0], this.p[1]) - this.surface.fRef;
	}

	get x(): number {
		return this.p[0];
	}
	get y(): number {
		return this.p[1];
	}
	/** Number of optimizer steps taken. */
	get steps(): number {
		return this.count - 1;
	}
	get lastLoss(): number {
		return this.loss[this.count - 1];
	}
}

export class Simulation {
	readonly runs: Run[];
	/** Global step counter (max over runs). */
	t = 0;
	private readonly box: { x0: number; x1: number; y0: number; y1: number };
	private readonly stochastic: boolean;

	constructor(
		readonly surface: Surface,
		private start: readonly [number, number],
		configs: readonly RunConfig[],
		readonly maxSteps: number,
		noise: number,
		seed: number
	) {
		this.stochastic = noise > 0;
		this.runs = configs.map((c, i) => new Run(c, surface, maxSteps, noise, deriveSeed(seed, i)));
		const d = surface.domain;
		const mx = (d.xMax - d.xMin) * surface.escapeMargin;
		const my = (d.yMax - d.yMin) * surface.escapeMargin;
		this.box = { x0: d.xMin - mx, x1: d.xMax + mx, y0: d.yMin - my, y1: d.yMax + my };
		this.reset();
	}

	reset(start: readonly [number, number] = this.start): void {
		this.start = start;
		this.t = 0;
		for (const r of this.runs) r.reset(start[0], start[1]);
	}

	get finished(): boolean {
		for (const r of this.runs) if (r.status === 'running') return false;
		return true;
	}

	/** Advance every running optimizer by up to k steps. Returns true while any run is active. */
	advance(k: number): boolean {
		const b = this.box;
		let active = false;
		for (const r of this.runs) {
			for (let j = 0; j < k && r.status === 'running'; j++) {
				const t = r.count; // 1-based step index
				r.opt.step(r.p, r.s, r.g, t, r.h, r.grad);
				const x = r.p[0];
				const y = r.p[1];
				if (!Number.isFinite(x) || !Number.isFinite(y)) {
					r.status = 'diverged';
					break;
				}
				r.record();
				if (x < b.x0 || x > b.x1 || y < b.y0 || y > b.y1) {
					r.status = 'escaped';
				} else if (!this.stochastic && Math.hypot(r.g[0], r.g[1]) < CONVERGED_GRAD_NORM) {
					r.status = 'converged';
				} else if (r.count > this.maxSteps) {
					r.status = 'done';
				}
			}
			if (r.steps > this.t) this.t = r.steps;
			if (r.status === 'running') active = true;
		}
		return active;
	}
}
