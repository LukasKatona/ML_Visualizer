<!--
	Loss gap f(θ_t) − f* vs. iteration for every run, hand-rolled SVG.
	Each series is decimated to at most MAX_PTS points (keeping the last point), so the cost per
	update is bounded no matter how many steps were taken. Updates are throttled by the parent.
-->
<script lang="ts">
	import { OPTIMIZERS } from '#lib/optimizers/optimizers.ts';
	import type { Simulation } from '#lib/sim/engine.ts';

	interface Props {
		sim: Simulation | null;
		/** Bumped by the parent when new steps are available. */
		version: number;
		logY: boolean;
		yLabel: string;
	}

	let { sim, version, logY, yLabel }: Props = $props();

	const MAX_PTS = 400;
	const FLOOR = 1e-16;
	const M = { l: 52, r: 12, t: 10, b: 28 };
	const H = 230;

	let width = $state(320);
	const iw = $derived(Math.max(10, width - M.l - M.r));
	const ih = H - M.t - M.b;

	interface Series {
		id: string;
		name: string;
		d: string;
	}

	const view = $derived.by(() => {
		void version;
		const runs = sim?.runs ?? [];
		const tMax = Math.max(10, sim?.t ?? 0);
		let lo = Infinity;
		let hi = -Infinity;
		for (const r of runs) {
			for (let i = 0; i < r.count; i++) {
				const v = logY ? Math.max(r.loss[i], FLOOR) : r.loss[i];
				if (v < lo) lo = v;
				if (v > hi) hi = v;
			}
		}
		if (!Number.isFinite(lo)) {
			lo = logY ? 1e-3 : 0;
			hi = 1;
		}
		let y0: number, y1: number;
		if (logY) {
			y0 = Math.floor(Math.log10(lo));
			y1 = Math.ceil(Math.log10(hi));
			if (y1 - y0 < 1) y1 = y0 + 1;
		} else {
			y0 = Math.min(0, lo);
			y1 = hi > y0 ? hi * 1.05 : y0 + 1;
		}
		const sx = (t: number) => M.l + (t / tMax) * iw;
		const sy = (v: number) => {
			const u = logY ? Math.log10(Math.max(v, FLOOR)) : v;
			return M.t + ih - ((u - y0) / (y1 - y0)) * ih;
		};

		const series: Series[] = runs.map((r) => {
			const n = r.count;
			const stride = Math.max(1, Math.ceil(n / MAX_PTS));
			let d = '';
			for (let i = 0; i < n; i += stride) {
				d += `${d ? 'L' : 'M'}${sx(i).toFixed(1)},${sy(r.loss[i]).toFixed(1)}`;
			}
			if ((n - 1) % stride !== 0 && n > 0) {
				d += `L${sx(n - 1).toFixed(1)},${sy(r.loss[n - 1]).toFixed(1)}`;
			}
			return { id: r.id, name: r.opt.name, d };
		});

		// ticks
		const yTicks: { y: number; label: string }[] = [];
		if (logY) {
			const step = Math.max(1, Math.ceil((y1 - y0) / 6));
			for (let k = y0; k <= y1; k += step) yTicks.push({ y: sy(10 ** k), label: `1e${k}` });
		} else {
			const raw = (y1 - y0) / 5;
			const p = 10 ** Math.floor(Math.log10(raw));
			const st = (raw / p < 2 ? 2 : raw / p < 5 ? 5 : 10) * p;
			for (let v = Math.ceil(y0 / st) * st; v <= y1; v += st)
				yTicks.push({ y: sy(v), label: Number(v.toPrecision(3)).toString() });
		}
		const xTicks: { x: number; label: string }[] = [];
		const raw = tMax / 5;
		const p = 10 ** Math.floor(Math.log10(raw));
		const st = (raw / p < 2 ? 2 : raw / p < 5 ? 5 : 10) * p;
		for (let t = 0; t <= tMax; t += st) xTicks.push({ x: sx(t), label: String(t) });

		return { series, yTicks, xTicks, tMax };
	});

	// ---------- hover ----------
	let hoverT = $state<number | null>(null);

	function onMove(e: PointerEvent) {
		const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
		const x = e.clientX - r.left - M.l;
		if (x < 0 || x > iw) {
			hoverT = null;
			return;
		}
		hoverT = Math.round((x / iw) * view.tMax);
	}

	const tip = $derived.by(() => {
		void version;
		if (hoverT === null || !sim) return null;
		const t = hoverT;
		const rows = sim.runs
			.filter((r) => t < r.count)
			.map((r) => ({ id: r.id, name: OPTIMIZERS[r.id].name, v: r.loss[t] }))
			.sort((a, b) => a.v - b.v);
		return { t, x: M.l + (t / view.tMax) * iw, rows };
	});
</script>

<div class="chart" bind:clientWidth={width}>
	{#if sim && sim.runs.length > 1}
		<ul class="legend" aria-label="Legend">
			{#each sim.runs as r (r.id)}
				<li><span class="sw" style:background={`var(--c-${r.id})`}></span>{r.opt.name}</li>
			{/each}
		</ul>
	{/if}
	<svg
		{width}
		height={H}
		role="img"
		aria-label={`Loss ${yLabel} versus iteration${logY ? ' (log scale)' : ''}. Exact values are in the optimizer table.`}
		onpointermove={onMove}
		onpointerleave={() => (hoverT = null)}
	>
		<g class="grid">
			{#each view.yTicks as t (t.label)}
				<line x1={M.l} x2={M.l + iw} y1={t.y} y2={t.y} />
				<text x={M.l - 6} y={t.y} dy="0.32em" text-anchor="end">{t.label}</text>
			{/each}
			{#each view.xTicks as t (t.label)}
				<text x={t.x} y={H - 8} text-anchor="middle">{t.label}</text>
			{/each}
			<line class="axis" x1={M.l} x2={M.l + iw} y1={M.t + ih} y2={M.t + ih} />
		</g>
		<g class="lines">
			{#each view.series as s (s.id)}
				<path d={s.d} style:stroke={`var(--c-${s.id})`} />
			{/each}
		</g>
		{#if tip}
			<line class="cross" x1={tip.x} x2={tip.x} y1={M.t} y2={M.t + ih} />
		{/if}
	</svg>
	{#if tip && tip.rows.length}
		<div class="tip" style:left={`${Math.min(tip.x + 12, width - 180)}px`} aria-hidden="true">
			<div class="tt">t = {tip.t}</div>
			{#each tip.rows as r (r.id)}
				<div class="tr">
					<span class="sw" style:background={`var(--c-${r.id})`}></span>
					<span>{r.name}</span>
					<span class="num">{r.v.toExponential(2)}</span>
				</div>
			{/each}
		</div>
	{/if}
</div>

<style>
	.chart {
		position: relative;
		width: 100%;
		min-width: 0;
		overflow: hidden;
	}
	svg {
		display: block;
		overflow: visible;
		touch-action: pan-y;
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		margin: 0 0 6px;
		padding: 0;
		list-style: none;
		font-size: 12.5px;
		color: var(--text-2);
	}
	.legend li {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
	.sw {
		width: 10px;
		height: 10px;
		border-radius: 2px;
		flex: none;
	}
	.grid line {
		stroke: var(--border);
		stroke-width: 1;
	}
	.grid .axis {
		stroke: var(--border-strong);
	}
	.grid text {
		font-family: var(--font-mono);
		font-size: 10.5px;
		fill: var(--muted);
	}
	.lines path {
		fill: none;
		stroke-width: 2;
		stroke-linejoin: round;
		stroke-linecap: round;
	}
	.cross {
		stroke: var(--text-2);
		stroke-width: 1;
		stroke-dasharray: 3 3;
	}
	.tip {
		position: absolute;
		top: 34px;
		min-width: 168px;
		padding: 6px 8px;
		font-size: 12px;
		background: var(--surface);
		border: 1px solid var(--border-strong);
		border-radius: 6px;
		box-shadow: var(--shadow);
		pointer-events: none;
	}
	.tt {
		font-family: var(--font-mono);
		color: var(--muted);
		margin-bottom: 2px;
	}
	.tr {
		display: grid;
		grid-template-columns: 10px 1fr auto;
		align-items: center;
		gap: 6px;
		color: var(--text);
	}
</style>
