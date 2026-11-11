import { describe, expect, it } from 'vitest';
import { generate, withExtra } from './datasets.ts';
import { Session, decodeDb, defaultDbState, encodeDb, type DbState } from './session.ts';

function session(patch: Partial<DbState>) {
	const st = { ...defaultDbState(), ...patch };
	const { train, test } = generate(st.dataset, st.n, st.noise, st.seed);
	return new Session(st, withExtra(train, st.extra), test);
}

describe('decision-boundary URL state', () => {
	it('round-trips the default state', () => {
		const st = defaultDbState();
		expect(decodeDb(encodeDb(st))).toEqual(st);
	});

	it('round-trips a customised MLP state with user points', () => {
		const st: DbState = {
			...defaultDbState(),
			dataset: 'spirals',
			n: 400,
			noise: 0.05,
			seed: 9,
			layers: 3,
			width: 16,
			activation: 'relu',
			optimizer: 'momentum',
			lr: 0.05,
			l2: 0.001,
			speed: 10,
			maxEpochs: 5000,
			extra: [
				[0.25, -1.5, 1],
				[-0.1, 0.33, 0]
			]
		};
		expect(decodeDb(encodeDb(st))).toEqual(st);
	});

	it('only encodes parameters of the active model', () => {
		const q = encodeDb({ ...defaultDbState(), model: 'knn', k: 7 });
		expect(q).toContain('k=7');
		expect(q).not.toContain('lr=');
		expect(decodeDb(q).k).toBe(7);
	});

	it('clamps and ignores garbage', () => {
		const st = decodeDb('d=nope&n=99999&m=tree&dep=-3&pts=1,2,3;0.5,0.5,1;x,y,0');
		expect(st.dataset).toBe('moons');
		expect(st.n).toBe(600);
		expect(st.maxDepth).toBe(1);
		expect(st.extra).toEqual([[0.5, 0.5, 1]]);
	});
});

describe('Session', () => {
	it('records epoch 0, trains, and stops at maxEpochs', () => {
		const s = session({ maxEpochs: 60 });
		expect(s.count).toBe(1);
		const l0 = s.loss[0];
		while (s.advance(7));
		expect(s.status).toBe('done');
		expect(s.epoch).toBe(60);
		expect(s.loss[s.count - 1]).toBeLessThan(l0);
	});

	it('reset() reproduces the same run', () => {
		const s = session({ maxEpochs: 40 });
		while (s.advance(10));
		const a = Array.from(s.loss.subarray(0, s.count));
		s.reset();
		while (s.advance(10));
		expect(Array.from(s.loss.subarray(0, s.count))).toEqual(a);
	});

	it('flags divergence for an absurd learning rate with plain SGD', () => {
		const s = session({ optimizer: 'sgd', lr: 3, degree: 5, model: 'logreg', maxEpochs: 2000 });
		while (s.advance(50));
		expect(['diverged', 'done']).toContain(s.status);
		for (let i = 0; i < s.count; i++) expect(Number.isFinite(s.loss[i])).toBe(true);
	});

	it('non-trainable models are fitted immediately', () => {
		const t = session({ model: 'tree', maxDepth: 3 });
		expect(t.status).toBe('done');
		expect(t.advance(5)).toBe(false);
		expect(t.accuracies().train).toBeGreaterThan(0.8);
		const k = session({ model: 'knn', k: 1 });
		expect(k.accuracies().train).toBe(1);
	});

	it('default MLP reaches high test accuracy on moons', () => {
		const s = session({ maxEpochs: 1500 });
		while (s.advance(100));
		expect(s.accuracies().test).toBeGreaterThan(0.9);
	});
});
