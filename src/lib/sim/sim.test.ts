import { describe, expect, it } from 'vitest';
import { SURFACES, SURFACE_IDS } from '../math/surfaces.ts';
import { OPTIMIZER_IDS } from '../optimizers/optimizers.ts';
import { Simulation } from './engine.ts';
import { decode, defaultState, encode, runsFor } from './urlState.ts';

describe('Simulation', () => {
	it('records the start point and stops at maxSteps', () => {
		const st = defaultState('quadratic');
		const sim = new Simulation(
			SURFACES.quadratic,
			st.start,
			runsFor('quadratic', ['sgd']),
			50,
			0,
			1
		);
		expect(sim.runs[0].count).toBe(1);
		while (sim.advance(7));
		const r = sim.runs[0];
		expect(r.steps).toBeLessThanOrEqual(50);
		expect(['done', 'converged']).toContain(r.status);
		expect(r.traj[0]).toBe(st.start[0]);
		expect(r.loss[0]).toBeCloseTo(SURFACES.quadratic.f(...st.start), 12);
	});

	it('flags divergence / escape instead of producing NaN', () => {
		const runs = [{ id: 'sgd' as const, h: { lr: 1 } }];
		const sim = new Simulation(SURFACES.quadratic, [1, 1], runs, 1000, 0, 1);
		while (sim.advance(50));
		expect(['diverged', 'escaped']).toContain(sim.runs[0].status);
		for (let i = 0; i < sim.runs[0].count; i++)
			expect(Number.isFinite(sim.runs[0].loss[i])).toBe(true);
	});

	it('stochastic runs are reproducible for a seed and differ across seeds', () => {
		const s = SURFACES.noisy;
		const mk = (seed: number) => {
			const sim = new Simulation(
				s,
				s.defaultStart,
				runsFor('noisy', ['sgd', 'adam']),
				200,
				2,
				seed
			);
			while (sim.advance(20));
			return sim.runs.map((r) => [r.x, r.y]);
		};
		expect(mk(5)).toEqual(mk(5));
		expect(mk(5)).not.toEqual(mk(6));
	});

	it('reset() replays an identical stochastic trajectory', () => {
		const s = SURFACES.noisy;
		const sim = new Simulation(s, s.defaultStart, runsFor('noisy', ['momentum']), 100, 2, 3);
		while (sim.advance(10));
		const a = [sim.runs[0].x, sim.runs[0].y];
		sim.reset();
		while (sim.advance(10));
		expect([sim.runs[0].x, sim.runs[0].y]).toEqual(a);
	});

	it('every optimizer stays finite with surface defaults on every surface', () => {
		for (const sid of SURFACE_IDS) {
			const s = SURFACES[sid];
			const sim = new Simulation(s, s.defaultStart, runsFor(sid, OPTIMIZER_IDS), 1000, s.noise, 1);
			while (sim.advance(100));
			for (const r of sim.runs) expect(r.status, `${r.id} on ${sid}`).not.toBe('diverged');
		}
	});
});

describe('URL state', () => {
	it('round-trips the default state', () => {
		const st = defaultState();
		expect(decode(encode(st))).toEqual(st);
	});

	it('round-trips a customised state', () => {
		const st = defaultState('beale');
		st.start = [1.25, -0.5];
		st.runs = runsFor('beale', ['adamw', 'nesterov']);
		st.runs[0].h.lr = 0.0123;
		st.runs[0].h.wd = 0.05;
		st.selected = 'nesterov';
		st.maxSteps = 2500;
		st.speed = 10;
		st.noise = 0.5;
		st.seed = 42;
		st.logY = false;
		expect(decode(encode(st))).toEqual(st);
	});

	it('falls back to defaults on garbage and clamps out-of-range values', () => {
		const st = decode('?s=nope&p=abc&o=adam:lr=99:zz=1,bogus,adam:lr=1&n=-5&v=1e9&ly=0');
		expect(st.surface).toBe('rosenbrock');
		expect(st.start).toEqual([...SURFACES.rosenbrock.defaultStart]);
		expect(st.runs).toHaveLength(1);
		expect(st.runs[0].h.lr).toBe(1);
		expect(st.maxSteps).toBe(10);
		expect(st.speed).toBe(200);
		expect(st.logY).toBe(false);
	});

	it('produces a readable query string', () => {
		const q = encode(defaultState('quadratic'));
		expect(q).toMatch(/^s=quadratic&p=-5,2&o=sgd:lr=0\.04,/);
		expect(q).not.toMatch(/%/);
	});
});
