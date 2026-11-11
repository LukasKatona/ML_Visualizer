<!--
	Generic multi-series line chart (SVG) over an integer x axis (iterations/epochs).
	Series are decimated to MAX_PTS points; a crosshair tooltip shows exact values.
	Exactly one y-scale: put measures with different units in separate charts.
-->
<script lang="ts">
	export interface LineSeries {
		id: string;
		label: string;
		/** CSS colour (e.g. 'var(--series-1)'). */
		color: string;
		data: Float64Array;
		dashed?: boolean;
	}

	interface Props {
		series: LineSeries[];
		/** Number of valid points in every series. */
		count: number;
		/** Bump to redraw (the data arrays are mutated in place). */
		version: number;
		logY?: boolean;
		/** Fixed linear y-domain; otherwise auto. */
		yDomain?: [number, number];
		yFormat?: (v: number) => string;
		label: string;
		height?: number;
	}

	let {
		series,
		count,
		version,
		logY = false,
		yDomain,
		yFormat = (v) => String(Number(v.toPrecision(3))),
		label,
		height = 180
	}: Props = $props();

	const MAX_PTS = 400;
	const FLOOR = 1e-16;
	const M = { l: 48, r: 10, t: 8, b: 24 };

	let width = $state(320);
	const iw = $derived(Math.max(10, width - M.l - M.r));
	const ih = $derived(height - M.t - M.b);

	function niceStep(span: number, target: number) {
		const raw = span / target;
		const p = 10 ** Math.floor(Math.log10(raw));
		const m = raw / p;
		return (m < 2 ? 2 : m < 5 ? 5 : 10) * p;
	}

	const view = $derived.by(() => {
		void version;
		const n = Math.max(1, count);
		const xMax = Math.max(10, n - 1);
		let y0: number;
		let y1: number;
		if (yDomain) [y0, y1] = yDomain;
		else {
			let lo = Infinity;
			let hi = -Infinity;
			for (const s of series)
				for (let i = 0; i < count; i++) {
					const v = logY ? Math.max(s.data[i], FLOOR) : s.data[i];
					if (v < lo) lo = v;
					if (v > hi) hi = v;
				}
			if (!Number.isFinite(lo)) [lo, hi] = logY ? [1e-3, 1] : [0, 1];
			if (logY) {
				y0 = Math.floor(Math.log10(lo));
				y1 = Math.ceil(Math.log10(hi));
				if (y1 - y0 < 1) y1 = y0 + 1;
			} else {
				const pad = (hi - lo) * 0.05 || 0.5;
				y0 = lo - pad;
				y1 = hi + pad;
			}
		}
		const sx = (i: number) => M.l + (i / xMax) * iw;
		const sy = (v: number) => {
			const u = logY && !yDomain ? Math.log10(Math.max(v, FLOOR)) : v;
			return M.t + ih - ((u - y0) / (y1 - y0)) * ih;
		};
		const stride = Math.max(1, Math.ceil(count / MAX_PTS));
		const paths = series.map((s) => {
			let d = '';
			for (let i = 0; i < count; i += stride)
				d += `${d ? 'L' : 'M'}${sx(i).toFixed(1)},${sy(s.data[i]).toFixed(1)}`;
			if (count > 0 && (count - 1) % stride !== 0)
				d += `L${sx(count - 1).toFixed(1)},${sy(s.data[count - 1]).toFixed(1)}`;
			return { ...s, d };
		});
		const yTicks: { y: number; label: string }[] = [];
		if (logY && !yDomain) {
			const st = Math.max(1, Math.ceil((y1 - y0) / 5));
			for (let k = y0; k <= y1; k += st) yTicks.push({ y: sy(10 ** k), label: `1e${k}` });
		} else {
			const st = niceStep(y1 - y0, 4);
			for (let v = Math.ceil(y0 / st) * st; v <= y1 + 1e-9; v += st)
				yTicks.push({ y: sy(v), label: yFormat(v) });
		}
		const xTicks: { x: number; label: string }[] = [];
		const xs = niceStep(xMax, 5);
		for (let t = 0; t <= xMax; t += xs) xTicks.push({ x: sx(t), label: String(t) });
		return { paths, yTicks, xTicks, xMax };
	});

	let hoverI = $state<number | null>(null);

	function onMove(e: PointerEvent) {
		const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
		const x = e.clientX - r.left - M.l;
		hoverI = x < 0 || x > iw ? null : Math.round((x / iw) * view.xMax);
	}

	const tip = $derived.by(() => {
		void version;
		if (hoverI === null || hoverI >= count) return null;
		const i = hoverI;
		return {
			i,
			x: M.l + (i / view.xMax) * iw,
			rows: series.map((s) => ({ ...s, v: s.data[i] }))
		};
	});
</script>

<div class="chart" bind:clientWidth={width}>
	{#if series.length > 1}
		<ul class="legend" aria-label="Legend">
			{#each series as s (s.id)}
				<li>
					<svg width="18" height="8" aria-hidden="true"
						><line
							x1="1"
							x2="17"
							y1="4"
							y2="4"
							stroke={s.color}
							stroke-width="2"
							stroke-dasharray={s.dashed ? '4 3' : undefined}
						/></svg
					>{s.label}
				</li>
			{/each}
		</ul>
	{/if}
	<svg
		{width}
		{height}
		role="img"
		aria-label={label}
		onpointermove={onMove}
		onpointerleave={() => (hoverI = null)}
	>
		<g class="grid">
			{#each view.yTicks as t (t.label)}
				<line x1={M.l} x2={M.l + iw} y1={t.y} y2={t.y} />
				<text x={M.l - 6} y={t.y} dy="0.32em" text-anchor="end">{t.label}</text>
			{/each}
			{#each view.xTicks as t (t.label)}
				<text x={t.x} y={height - 6} text-anchor="middle">{t.label}</text>
			{/each}
			<line class="axis" x1={M.l} x2={M.l + iw} y1={M.t + ih} y2={M.t + ih} />
		</g>
		{#each view.paths as p (p.id)}
			<path
				class="line"
				d={p.d}
				style:stroke={p.color}
				stroke-dasharray={p.dashed ? '5 4' : undefined}
			/>
		{/each}
		{#if tip}
			<line class="cross" x1={tip.x} x2={tip.x} y1={M.t} y2={M.t + ih} />
		{/if}
	</svg>
	{#if tip}
		<div class="tip" style:left={`${Math.min(tip.x + 12, width - 150)}px`} aria-hidden="true">
			<div class="tt">epoch {tip.i}</div>
			{#each tip.rows as r (r.id)}
				<div class="tr">
					<span>{r.label}</span><span class="num">{yFormat(r.v)}</span>
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
		touch-action: pan-y;
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		margin: 0 0 4px;
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
	.grid line {
		stroke: var(--border);
	}
	.grid .axis {
		stroke: var(--border-strong);
	}
	.grid text {
		font-family: var(--font-mono);
		font-size: 10.5px;
		fill: var(--muted);
	}
	.line {
		fill: none;
		stroke-width: 2;
		stroke-linejoin: round;
		stroke-linecap: round;
	}
	.cross {
		stroke: var(--text-2);
		stroke-dasharray: 3 3;
	}
	.tip {
		position: absolute;
		top: 26px;
		min-width: 138px;
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
	}
	.tr {
		display: flex;
		justify-content: space-between;
		gap: 12px;
	}
</style>
