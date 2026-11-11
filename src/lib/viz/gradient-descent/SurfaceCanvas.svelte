<!--
	Three stacked canvases:
	  base   – heatmap, isolines, ticks, minima (repainted only on surface/size/theme change)
	  trails – trajectories, drawn incrementally (only new segments each frame)
	  top    – start marker, current heads + labels, keyboard cursor (cleared every frame; cheap)
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { isolines, makeRamp, paintField, sampleField, type Field } from '#lib/math/field.ts';
	import type { Surface } from '#lib/math/surfaces.ts';
	import { OPTIMIZERS, type OptimizerId } from '#lib/optimizers/optimizers.ts';
	import type { Simulation } from '#lib/sim/engine.ts';
	import { fmt } from '#lib/sim/urlState.ts';
	import { theme } from '#lib/theme.svelte.ts';

	interface Props {
		surface: Surface;
		sim: Simulation | null;
		start: readonly [number, number];
		selected: OptimizerId;
		onpick: (x: number, y: number) => void;
	}

	let { surface, sim, start, selected, onpick }: Props = $props();

	let wrap: HTMLDivElement;
	let baseCv: HTMLCanvasElement;
	let trailCv: HTMLCanvasElement;
	let topCv: HTMLCanvasElement;
	let W = $state(0);
	let H = $state(0);
	let dpr = 1;

	let hover = $state<{ x: number; y: number; f: number } | null>(null);
	let focused = $state(false);
	let cursor = $state<[number, number]>([0, 0]);

	// colours resolved from CSS custom properties on theme change
	let colors: Record<string, string> = {};
	let surfaceColor = '#fff';
	let textColor = '#000';

	/** Per-run number of trajectory points already drawn on the trails layer. */
	let drawn = new Int32Array(0);
	let drawnFor: Simulation | null = null;

	const dom = $derived(surface.domain);
	const aspect = $derived((dom.xMax - dom.xMin) / (dom.yMax - dom.yMin));

	const px = (x: number) => ((x - dom.xMin) / (dom.xMax - dom.xMin)) * W;
	const py = (y: number) => ((dom.yMax - y) / (dom.yMax - dom.yMin)) * H;
	const wx = (u: number) => dom.xMin + (u / W) * (dom.xMax - dom.xMin);
	const wy = (v: number) => dom.yMax - (v / H) * (dom.yMax - dom.yMin);

	function readColors() {
		const cs = getComputedStyle(document.documentElement);
		colors = {};
		for (const id of Object.keys(OPTIMIZERS)) colors[id] = cs.getPropertyValue(`--c-${id}`).trim();
		surfaceColor = cs.getPropertyValue('--surface').trim();
		textColor = cs.getPropertyValue('--text').trim();
	}

	function ctx2d(cv: HTMLCanvasElement) {
		const c = cv.getContext('2d')!;
		c.setTransform(dpr, 0, 0, dpr, 0, 0);
		return c;
	}

	function niceStep(span: number) {
		const raw = span / 5;
		const p = 10 ** Math.floor(Math.log10(raw));
		const m = raw / p;
		return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
	}

	let field: Field | null = null;
	let fieldKey = '';

	function paintBase() {
		if (!W || !H) return;
		const ramp = makeRamp(theme.current);
		// sample at ~half device resolution (capped), then upscale with smoothing
		const nx = Math.min(480, Math.max(64, Math.round(W * 0.6)));
		const ny = Math.max(32, Math.round(nx / aspect));
		const key = `${surface.id}:${nx}x${ny}`;
		if (key !== fieldKey) {
			field = sampleField(surface, nx, ny);
			fieldKey = key;
		}
		const f = field!;
		const off = new OffscreenCanvas(nx, ny);
		const octx = off.getContext('2d')!;
		const img = octx.createImageData(nx, ny);
		paintField(f, ramp, img);
		octx.putImageData(img, 0, 0);

		const c = ctx2d(baseCv);
		c.imageSmoothingEnabled = true;
		c.imageSmoothingQuality = 'high';
		c.clearRect(0, 0, W, H);
		c.drawImage(off, 0, 0, W, H);

		// isolines, evenly spaced in log-loss
		const levels = Array.from({ length: 15 }, (_, k) => (k + 1) / 16);
		const segs = isolines(f, levels);
		const sx = W / (f.nx - 1);
		const sy = H / (f.ny - 1);
		c.strokeStyle = ramp.iso;
		c.lineWidth = 1;
		c.beginPath();
		for (let k = 0; k < segs.length; k += 4) {
			c.moveTo(segs[k] * sx, segs[k + 1] * sy);
			c.lineTo(segs[k + 2] * sx, segs[k + 3] * sy);
		}
		c.stroke();

		// ticks (text with a surface halo so they stay legible over any colour)
		c.font = '11px ui-monospace, Menlo, monospace';
		c.lineJoin = 'round';
		c.lineWidth = 3;
		c.strokeStyle = surfaceColor;
		c.fillStyle = textColor;
		const label = (t: string, u: number, v: number) => {
			c.strokeText(t, u, v);
			c.fillText(t, u, v);
		};
		const stx = niceStep(dom.xMax - dom.xMin);
		const sty = niceStep(dom.yMax - dom.yMin);
		c.textAlign = 'center';
		c.textBaseline = 'bottom';
		for (let x = Math.ceil(dom.xMin / stx) * stx; x <= dom.xMax; x += stx) {
			if (Math.abs(x - dom.xMin) < stx * 0.3) continue;
			c.fillRect(px(x) - 0.5, H - 4, 1, 4);
			label(fmt(x), px(x), H - 6);
		}
		c.textAlign = 'left';
		c.textBaseline = 'middle';
		for (let y = Math.ceil(dom.yMin / sty) * sty; y <= dom.yMax; y += sty) {
			if (Math.abs(y - dom.yMin) < sty * 0.3) continue;
			c.fillRect(0, py(y) - 0.5, 4, 1);
			label(fmt(y), 7, py(y));
		}

		// known minima
		c.strokeStyle = textColor;
		c.globalAlpha = 0.7;
		c.lineWidth = 1.5;
		for (const [mx, my] of surface.minima) {
			const u = px(mx);
			const v = py(my);
			c.beginPath();
			c.moveTo(u - 5, v - 5);
			c.lineTo(u + 5, v + 5);
			c.moveTo(u + 5, v - 5);
			c.lineTo(u - 5, v + 5);
			c.stroke();
		}
		c.globalAlpha = 1;
	}

	function strokeRange(
		c: CanvasRenderingContext2D,
		tr: Float64Array,
		from: number,
		to: number,
		color: string
	) {
		c.beginPath();
		c.moveTo(px(tr[2 * from]), py(tr[2 * from + 1]));
		for (let i = from + 1; i < to; i++) c.lineTo(px(tr[2 * i]), py(tr[2 * i + 1]));
		c.lineCap = 'butt';
		c.lineWidth = 4;
		c.strokeStyle = surfaceColor;
		c.globalAlpha = 0.55;
		c.stroke();
		c.globalAlpha = 1;
		c.lineCap = 'round';
		c.lineWidth = 2;
		c.strokeStyle = color;
		c.stroke();
	}

	/** Clear and redraw all trails from scratch (after reset, resize or theme change). */
	export function redraw(): void {
		if (!W || !trailCv) return;
		const c = ctx2d(trailCv);
		c.clearRect(0, 0, W, H);
		drawn = new Int32Array(sim ? sim.runs.length : 0);
		drawnFor = sim;
		frame();
	}

	/** Draw new trajectory segments and the per-frame overlay. Allocation-free on the trail path. */
	export function frame(): void {
		if (!W || !trailCv) return;
		if (drawnFor !== sim) return redraw();
		const tc = ctx2d(trailCv);
		tc.lineJoin = 'round';
		if (sim) {
			for (let r = 0; r < sim.runs.length; r++) {
				const run = sim.runs[r];
				const from = Math.max(0, drawn[r] - 1);
				if (run.count - from >= 2) strokeRange(tc, run.traj, from, run.count, colors[run.id]);
				drawn[r] = run.count;
			}
		}
		drawTop();
	}

	function drawTop() {
		const c = ctx2d(topCv);
		c.clearRect(0, 0, W, H);

		// start marker
		const su = px(start[0]);
		const sv = py(start[1]);
		c.lineWidth = 2;
		c.strokeStyle = surfaceColor;
		c.beginPath();
		c.arc(su, sv, 7, 0, 2 * Math.PI);
		c.stroke();
		c.lineWidth = 1.5;
		c.strokeStyle = textColor;
		c.beginPath();
		c.arc(su, sv, 6, 0, 2 * Math.PI);
		c.stroke();

		if (sim && sim.t > 0) {
			// heads, then labels with greedy vertical de-overlap
			c.font = '600 12px ui-sans-serif, system-ui, sans-serif';
			c.textBaseline = 'middle';
			c.lineJoin = 'round';
			const labels: { u: number; v: number; name: string }[] = [];
			for (const run of sim.runs) {
				const u = px(run.x);
				const v = py(run.y);
				if (u < -20 || u > W + 20 || v < -20 || v > H + 20) continue;
				const rad = run.id === selected ? 5.5 : 4;
				c.beginPath();
				c.arc(u, v, rad + 2, 0, 2 * Math.PI);
				c.fillStyle = surfaceColor;
				c.fill();
				c.beginPath();
				c.arc(u, v, rad, 0, 2 * Math.PI);
				c.fillStyle = colors[run.id];
				c.fill();
				labels.push({ u, v, name: run.opt.name });
			}
			labels.sort((a, b) => a.v - b.v);
			let lastV = -Infinity;
			for (const l of labels) {
				const v = Math.max(l.v, lastV + 14);
				lastV = v;
				const left = l.u > W - 90;
				c.textAlign = left ? 'right' : 'left';
				const tu = l.u + (left ? -10 : 10);
				c.lineWidth = 3;
				c.strokeStyle = surfaceColor;
				c.strokeText(l.name, tu, v);
				c.fillStyle = textColor;
				c.fillText(l.name, tu, v);
			}
		}

		if (focused) {
			const u = px(cursor[0]);
			const v = py(cursor[1]);
			c.strokeStyle = textColor;
			c.lineWidth = 1;
			c.setLineDash([3, 3]);
			c.beginPath();
			c.moveTo(u, 0);
			c.lineTo(u, H);
			c.moveTo(0, v);
			c.lineTo(W, v);
			c.stroke();
			c.setLineDash([]);
		}
	}

	function resize() {
		const r = wrap.getBoundingClientRect();
		dpr = Math.min(window.devicePixelRatio || 1, 2);
		W = Math.max(1, Math.round(r.width));
		H = Math.max(1, Math.round(r.height));
		for (const cv of [baseCv, trailCv, topCv]) {
			cv.width = Math.round(W * dpr);
			cv.height = Math.round(H * dpr);
		}
		paintBase();
		redraw();
	}

	onMount(() => {
		readColors();
		const ro = new ResizeObserver(() => resize());
		ro.observe(wrap);
		return () => ro.disconnect();
	});

	// repaint on surface or theme change
	$effect(() => {
		void surface;
		void theme.current;
		if (!W) return;
		readColors();
		paintBase();
		redraw();
	});

	// keep overlay in sync with start / selection / sim changes even when paused
	$effect(() => {
		void start;
		void selected;
		void sim;
		void focused;
		void cursor;
		if (W) frame();
	});

	function eventToWorld(e: PointerEvent): [number, number] {
		const r = wrap.getBoundingClientRect();
		return [wx(e.clientX - r.left), wy(e.clientY - r.top)];
	}

	function onPointerDown(e: PointerEvent) {
		if (e.button !== 0) return;
		const [x, y] = eventToWorld(e);
		cursor = [x, y];
		onpick(x, y);
	}

	function onPointerMove(e: PointerEvent) {
		const [x, y] = eventToWorld(e);
		hover = { x, y, f: surface.f(x, y) };
	}

	function onKeyDown(e: KeyboardEvent) {
		const fx = (dom.xMax - dom.xMin) * (e.shiftKey ? 0.05 : 0.01);
		const fy = (dom.yMax - dom.yMin) * (e.shiftKey ? 0.05 : 0.01);
		let [x, y] = cursor;
		switch (e.key) {
			case 'ArrowLeft':
				x -= fx;
				break;
			case 'ArrowRight':
				x += fx;
				break;
			case 'ArrowUp':
				y += fy;
				break;
			case 'ArrowDown':
				y -= fy;
				break;
			case 'Enter':
			case ' ':
				e.preventDefault();
				onpick(x, y);
				return;
			default:
				return;
		}
		e.preventDefault();
		x = Math.min(dom.xMax, Math.max(dom.xMin, x));
		y = Math.min(dom.yMax, Math.max(dom.yMin, y));
		cursor = [x, y];
		hover = { x, y, f: surface.f(x, y) };
	}
</script>

<figure class="surface">
	<!-- A 2D point picker with its own arrow-key handling: role="application" is the correct ARIA
	     role, but Svelte's a11y lint does not count it as interactive. -->
	<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
	<div
		bind:this={wrap}
		class="plot"
		style:aspect-ratio={aspect}
		style:--ar={aspect}
		role="application"
		aria-label={`${surface.name} loss surface. Click, or focus and use arrow keys then Enter, to set the start point.`}
		aria-describedby="surface-help"
		tabindex="0"
		onpointerdown={onPointerDown}
		onpointermove={onPointerMove}
		onpointerleave={() => (hover = null)}
		onkeydown={onKeyDown}
		onfocus={() => {
			cursor = [start[0], start[1]];
			focused = true;
		}}
		onblur={() => (focused = false)}
	>
		<canvas bind:this={baseCv}></canvas>
		<canvas bind:this={trailCv}></canvas>
		<canvas bind:this={topCv}></canvas>
		<div class="readout num" aria-hidden="true">
			{#if hover}
				x {fmt(hover.x)} · y {fmt(hover.y)} · f {fmt(hover.f)}
			{:else}
				start ({fmt(start[0])}, {fmt(start[1])})
			{/if}
		</div>
	</div>
	<figcaption id="surface-help" class="visually-hidden">
		Arrow keys move the cursor (Shift for larger steps); Enter sets the start point.
	</figcaption>
</figure>

<style>
	.surface {
		margin: 0;
	}
	.plot {
		position: relative;
		width: min(100%, calc(72vh * var(--ar)));
		margin: 0 auto;
		border: 1px solid var(--border);
		border-radius: var(--radius);
		overflow: hidden;
		cursor: crosshair;
		touch-action: manipulation;
		background: var(--surface);
	}
	canvas {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		display: block;
	}
	.readout {
		position: absolute;
		right: 8px;
		top: 8px;
		padding: 3px 8px;
		font-size: 11.5px;
		border-radius: 6px;
		background: color-mix(in srgb, var(--surface) 88%, transparent);
		border: 1px solid var(--border);
		color: var(--text-2);
		pointer-events: none;
	}
</style>
