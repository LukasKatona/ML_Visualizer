/**
 * Small seeded PRNG (mulberry32) with an allocation-free Gaussian sampler (Box–Muller,
 * caching the second variate). Seeded so that a shared URL reproduces a stochastic run exactly.
 */
export class Rng {
	private s: number;
	private spare = 0;
	private hasSpare = false;

	constructor(seed: number) {
		this.s = seed >>> 0;
	}

	reseed(seed: number): void {
		this.s = seed >>> 0;
		this.hasSpare = false;
	}

	/** Uniform in [0, 1). */
	uniform(): number {
		let t = (this.s = (this.s + 0x6d2b79f5) >>> 0);
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	}

	/** Standard normal N(0, 1). */
	normal(): number {
		if (this.hasSpare) {
			this.hasSpare = false;
			return this.spare;
		}
		let u = 0;
		while (u === 0) u = this.uniform();
		const v = this.uniform();
		const r = Math.sqrt(-2 * Math.log(u));
		const th = 2 * Math.PI * v;
		this.spare = r * Math.sin(th);
		this.hasSpare = true;
		return r * Math.cos(th);
	}
}

/** Derive independent per-run seeds from a base seed (golden-ratio hashing). */
export function deriveSeed(base: number, index: number): number {
	let h = (base ^ Math.imul(index + 1, 0x9e3779b9)) >>> 0;
	h = Math.imul(h ^ (h >>> 16), 0x85ebca6b) >>> 0;
	h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
	return (h ^ (h >>> 16)) >>> 0;
}
