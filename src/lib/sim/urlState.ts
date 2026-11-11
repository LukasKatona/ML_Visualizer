/**
 * Full visualizer state <-> compact, human-readable query string.
 *
 *   ?s=rosenbrock&p=-1.5,2.5&o=adam:lr=0.05:b1=0.9:b2=0.999:eps=1e-8,sgd:lr=0.0004
 *    &sel=adam&n=1000&v=3&sig=0&seed=1&ly=1
 *
 * Decoding is total: anything missing or invalid falls back to defaults (and values are
 * clamped to their slider ranges), so a hand-edited link never breaks the page.
 */
import { SURFACES, isSurfaceId, type SurfaceId } from '../math/surfaces.ts';
import {
	HYPER_META,
	OPTIMIZERS,
	defaultHyper,
	isOptimizerId,
	type Hyper,
	type HyperKey,
	type OptimizerId
} from '../optimizers/optimizers.ts';
import type { RunConfig } from './engine.ts';

export interface VizState {
	surface: SurfaceId;
	start: [number, number];
	runs: RunConfig[];
	selected: OptimizerId;
	maxSteps: number;
	/** Optimizer steps per animation frame. */
	speed: number;
	noise: number;
	seed: number;
	logY: boolean;
}

export const LIMITS = {
	maxSteps: { min: 10, max: 10_000 },
	speed: { min: 1, max: 200 },
	noise: { min: 0, max: 50 }
} as const;

export const DEFAULT_SURFACE: SurfaceId = 'rosenbrock';
export const DEFAULT_OPTIMIZERS: OptimizerId[] = ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam'];

const SHORT: Record<HyperKey, string> = {
	lr: 'lr',
	beta: 'b',
	beta1: 'b1',
	beta2: 'b2',
	eps: 'eps',
	wd: 'wd'
};
const LONG = Object.fromEntries(Object.entries(SHORT).map(([k, v]) => [v, k])) as Record<
	string,
	HyperKey
>;

export function runsFor(surface: SurfaceId, ids: readonly OptimizerId[]): RunConfig[] {
	const lr = SURFACES[surface].lr;
	return ids.map((id) => ({ id, h: defaultHyper(id, lr) }));
}

export function defaultState(surface: SurfaceId = DEFAULT_SURFACE): VizState {
	const s = SURFACES[surface];
	return {
		surface,
		start: [s.defaultStart[0], s.defaultStart[1]],
		runs: runsFor(surface, DEFAULT_OPTIMIZERS),
		selected: 'adam',
		maxSteps: 1000,
		speed: 3,
		noise: s.noise,
		seed: 1,
		logY: true
	};
}

/** Shortest decimal that round-trips to 4 significant digits. */
export function fmt(n: number): string {
	return String(Number(n.toPrecision(4)));
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

function num(s: string | null): number | null {
	if (s === null || s.trim() === '') return null;
	const v = Number(s);
	return Number.isFinite(v) ? v : null;
}

export function encode(st: VizState): string {
	const runs = st.runs
		.map((r) => {
			const parts: string[] = [r.id];
			for (const k of OPTIMIZERS[r.id].hypers) {
				const v = r.h[k];
				if (v !== undefined) parts.push(`${SHORT[k]}=${fmt(v)}`);
			}
			return parts.join(':');
		})
		.join(',');
	const q: [string, string][] = [
		['s', st.surface],
		['p', `${fmt(st.start[0])},${fmt(st.start[1])}`],
		['o', runs],
		['sel', st.selected],
		['n', String(st.maxSteps)],
		['v', String(st.speed)],
		['sig', fmt(st.noise)],
		['seed', String(st.seed)],
		['ly', st.logY ? '1' : '0']
	];
	// Values only contain [A-Za-z0-9.,:=+-]: safe unencoded in a query, readable when shared.
	return q.map(([k, v]) => `${k}=${v}`).join('&');
}

function decodeRuns(raw: string, surface: SurfaceId): RunConfig[] {
	const out: RunConfig[] = [];
	const seen = new Set<OptimizerId>();
	for (const chunk of raw.split(',')) {
		const [id, ...kvs] = chunk.split(':');
		if (!isOptimizerId(id) || seen.has(id)) continue;
		seen.add(id);
		const h: Hyper = defaultHyper(id, SURFACES[surface].lr);
		const allowed = OPTIMIZERS[id].hypers;
		for (const kv of kvs) {
			const eq = kv.indexOf('=');
			if (eq < 0) continue;
			const key = LONG[kv.slice(0, eq)];
			const v = num(kv.slice(eq + 1));
			if (!key || v === null || !allowed.includes(key)) continue;
			const m = HYPER_META[key];
			h[key] = clamp(v, m.min, m.max);
		}
		out.push({ id, h });
	}
	return out;
}

export function decode(search: string): VizState {
	const q = new URLSearchParams(search.startsWith('?') ? search.slice(1) : search);
	const sRaw = q.get('s') ?? '';
	const surface = isSurfaceId(sRaw) ? sRaw : DEFAULT_SURFACE;
	const st = defaultState(surface);
	const surf = SURFACES[surface];

	const p = q.get('p')?.split(',');
	if (p?.length === 2) {
		const x = num(p[0]);
		const y = num(p[1]);
		const d = surf.domain;
		if (x !== null && y !== null) {
			st.start = [clamp(x, d.xMin, d.xMax), clamp(y, d.yMin, d.yMax)];
		}
	}

	const o = q.get('o');
	if (o !== null) st.runs = decodeRuns(o, surface);

	const sel = q.get('sel') ?? '';
	if (isOptimizerId(sel)) st.selected = sel;
	else if (st.runs.length && !st.runs.some((r) => r.id === st.selected))
		st.selected = st.runs[0].id;

	const n = num(q.get('n'));
	if (n !== null) st.maxSteps = Math.round(clamp(n, LIMITS.maxSteps.min, LIMITS.maxSteps.max));
	const v = num(q.get('v'));
	if (v !== null) st.speed = Math.round(clamp(v, LIMITS.speed.min, LIMITS.speed.max));
	const sig = num(q.get('sig'));
	if (sig !== null) st.noise = clamp(sig, LIMITS.noise.min, LIMITS.noise.max);
	const seed = num(q.get('seed'));
	if (seed !== null) st.seed = Math.abs(Math.trunc(seed)) >>> 0;
	const ly = q.get('ly');
	if (ly !== null) st.logY = ly !== '0';
	return st;
}
