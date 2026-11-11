<!--
	P(y = 1 | x) heatmap with the p = 0.5 decision boundary, plus the data.
	  field  – heatmap + boundary, repainted by the parent via paint() (low-res while training)
	  points – data markers; class 0 = circle, class 1 = square; test points hollow
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { isolines, makeDivergingRamp, paintField } from '#lib/math/field.ts';
	import { DOMAIN, type Points } from '#lib/ml/datasets.ts';
	import { hasNeighbours, hasSplits, type Predictor } from '#lib/ml/session.ts';
	import { theme } from '#lib/theme.svelte.ts';

	interface Props {
		predictor: Predictor | null;
		train: Points;
		test: Points;
		showTest: boolean;
		/** k-NN replay: training points with index ≥ stored are not in memory yet (drawn faded). */
		stored?: number | null;
		/** Class placed by a plain click / Enter (Shift places the other class). */
		brush: 0 | 1;
		onadd: (x: number, y: number, cls: 0 | 1) => void;
	}

	let { predictor, train, test, showTest, stored = null, brush, onadd }: Props = $props();

	let wrap: HTMLDivElement;
	let fieldCv: HTMLCanvasElement;
	let pointCv: HTMLCanvasElement;
	let hoverCv: HTMLCanvasElement;
	let W = 0;
	let H = 0;
	let dpr = 1;

	let focused = $state(false);
	let cursor = $state<[number, number]>([0, 0]);
	let hover = $state<{ x: number; y: number; p: number | null } | null>(null);

	let c0 = '#2a78d6';
	let c1 = '#e34948';
	let surfaceColor = '#fff';
	let textColor = '#000';

	const D = DOMAIN;
	const px = (x: number) => ((x - D.xMin) / (D.xMax - D.xMin)) * W;
	const py = (y: number) => ((D.yMax - y) / (D.yMax - D.yMin)) * H;
	const wx = (u: number) => D.xMin + (u / W) * (D.xMax - D.xMin);
	const wy = (v: number) => D.yMax - (v / H) * (D.yMax - D.yMin);

	function readColors() {
		const cs = getComputedStyle(document.documentElement);
		c0 = cs.getPropertyValue('--class-0').trim();
		c1 = cs.getPropertyValue('--class-1').trim();
		surfaceColor = cs.getPropertyValue('--surface').trim();
		textColor = cs.getPropertyValue('--text').trim();
	}

	function ctx(cv: HTMLCanvasElement) {
		const c = cv.getContext('2d')!;
		c.setTransform(dpr, 0, 0, dpr, 0, 0);
		return c;
	}

	// reusable grid buffers (reallocated only when the resolution changes)
	let grid = { nx: 0, ny: 0, t: new Float32Array(0) };
	let off: OffscreenCanvas | null = null;
	let img: ImageData | null = null;

	/** Evaluate the model on a grid and repaint. `fast` uses a coarse grid (while training). */
	export function paint(fast = false): void {
		if (!W || !fieldCv) return;
		const c = ctx(fieldCv);
		c.clearRect(0, 0, W, H);
		if (!predictor) return;
		const nx = fast ? 56 : Math.min(200, Math.max(64, Math.round(W / 3)));
		const ny = nx;
		if (grid.nx !== nx) {
			grid = { nx, ny, t: new Float32Array(nx * ny) };
			off = new OffscreenCanvas(nx, ny);
			img = off.getContext('2d')!.createImageData(nx, ny);
		}
		const t = grid.t;
		for (let j = 0; j < ny; j++) {
			const y = D.yMax - ((D.yMax - D.yMin) * (j + 0.5)) / ny;
			for (let i = 0; i < nx; i++) {
				const x = D.xMin + ((D.xMax - D.xMin) * (i + 0.5)) / nx;
				t[j * nx + i] = predictor.predict(x, y);
			}
		}
		const ramp = makeDivergingRamp(theme.current);
		paintField(grid, ramp, img!);
		off!.getContext('2d')!.putImageData(img!, 0, 0);
		c.imageSmoothingEnabled = true;
		c.imageSmoothingQuality = 'high';
		c.drawImage(off!, 0, 0, W, H);

		// decision boundary p = 0.5 (grid cell centres → pixels)
		const segs = isolines(grid, [0.5]);
		const sx = W / nx;
		const sy = H / ny;
		c.lineJoin = 'round';
		c.lineCap = 'round';
		c.beginPath();
		for (let k = 0; k < segs.length; k += 4) {
			c.moveTo((segs[k] + 0.5) * sx, (segs[k + 1] + 0.5) * sy);
			c.lineTo((segs[k + 2] + 0.5) * sx, (segs[k + 3] + 0.5) * sy);
		}
		c.strokeStyle = surfaceColor;
		c.lineWidth = 4;
		c.globalAlpha = 0.6;
		c.stroke();
		c.globalAlpha = 1;
		c.strokeStyle = ramp.iso;
		c.lineWidth = 1.75;
		c.stroke();

		// tree partition: every split as a thin segment, the newest one emphasised
		if (hasSplits(predictor) && predictor.splits.length) {
			const segs2 = predictor.splits;
			const last = segs2.length - 1;
			c.setLineDash([4, 3]);
			for (let i = 0; i <= last; i++) {
				const sp = segs2[i];
				const newest = i === last;
				c.beginPath();
				if (sp.feat === 0) {
					c.moveTo(px(sp.thr), py(sp.lo));
					c.lineTo(px(sp.thr), py(sp.hi));
				} else {
					c.moveTo(px(sp.lo), py(sp.thr));
					c.lineTo(px(sp.hi), py(sp.thr));
				}
				c.strokeStyle = textColor;
				c.globalAlpha = newest ? 0.95 : 0.35;
				c.lineWidth = newest ? 2 : 1;
				c.stroke();
			}
			c.setLineDash([]);
			c.globalAlpha = 1;
		}
	}

	// ---------- hover: k-NN neighbourhood ----------
	const nbr = new Int32Array(128);
	let hoverPt: [number, number] | null = null;
	let vote = $state<{ ones: number; k: number } | null>(null);

	function drawHover() {
		if (!W || !hoverCv) return;
		const c = ctx(hoverCv);
		c.clearRect(0, 0, W, H);
		vote = null;
		if (!hoverPt || !predictor || !hasNeighbours(predictor)) return;
		const m = predictor.neighbors(hoverPt[0], hoverPt[1], nbr);
		if (!m) return;
		const u = px(hoverPt[0]);
		const v = py(hoverPt[1]);
		const d = predictor.data;
		let ones = 0;
		// radius = distance to the k-th neighbour
		const far = nbr[m - 1];
		const r = Math.hypot(px(d.x[2 * far]) - u, py(d.x[2 * far + 1]) - v);
		c.beginPath();
		c.arc(u, v, r, 0, 2 * Math.PI);
		c.setLineDash([4, 4]);
		c.strokeStyle = textColor;
		c.globalAlpha = 0.7;
		c.lineWidth = 1.25;
		c.stroke();
		c.setLineDash([]);
		c.globalAlpha = 0.55;
		c.lineWidth = 1;
		c.beginPath();
		for (let i = 0; i < m; i++) {
			const s2 = nbr[i];
			ones += d.y[s2];
			c.moveTo(u, v);
			c.lineTo(px(d.x[2 * s2]), py(d.x[2 * s2 + 1]));
		}
		c.stroke();
		c.globalAlpha = 1;
		for (let i = 0; i < m; i++) {
			const s2 = nbr[i];
			c.beginPath();
			c.arc(px(d.x[2 * s2]), py(d.x[2 * s2 + 1]), 7, 0, 2 * Math.PI);
			c.lineWidth = 1.5;
			c.strokeStyle = textColor;
			c.stroke();
		}
		vote = { ones, k: m };
	}

	function marker(c: CanvasRenderingContext2D, u: number, v: number, cls: number, hollow: boolean) {
		const r = 4.2;
		c.beginPath();
		if (cls === 0) c.arc(u, v, r, 0, 2 * Math.PI);
		else c.rect(u - r * 0.88, v - r * 0.88, r * 1.76, r * 1.76);
		const col = cls === 0 ? c0 : c1;
		if (hollow) {
			c.lineWidth = 3.5;
			c.strokeStyle = surfaceColor;
			c.stroke();
			c.lineWidth = 1.8;
			c.strokeStyle = col;
			c.stroke();
		} else {
			c.lineWidth = 2;
			c.strokeStyle = surfaceColor;
			c.stroke();
			c.fillStyle = col;
			c.fill();
		}
	}

	function drawPoints() {
		if (!W || !pointCv) return;
		const c = ctx(pointCv);
		c.clearRect(0, 0, W, H);
		c.lineJoin = 'round';
		const limit = stored ?? train.n;
		for (let i = 0; i < train.n; i++) {
			c.globalAlpha = i < limit ? 1 : 0.22;
			marker(c, px(train.x[2 * i]), py(train.x[2 * i + 1]), train.y[i], false);
		}
		c.globalAlpha = 1;
		if (showTest)
			for (let i = 0; i < test.n; i++)
				marker(c, px(test.x[2 * i]), py(test.x[2 * i + 1]), test.y[i], true);
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
		for (const cv of [fieldCv, pointCv, hoverCv]) {
			cv.width = Math.round(W * dpr);
			cv.height = Math.round(H * dpr);
		}
		paint();
		drawPoints();
		drawHover();
	}

	onMount(() => {
		readColors();
		const ro = new ResizeObserver(() => resize());
		ro.observe(wrap);
		return () => ro.disconnect();
	});

	$effect(() => {
		void theme.current;
		void predictor;
		if (!W) return;
		readColors();
		paint();
		drawPoints();
	});

	$effect(() => {
		void train;
		void test;
		void showTest;
		void stored;
		void focused;
		void cursor;
		drawPoints();
	});

	function readout(h: { x: number; y: number; p: number | null }): string {
		const parts = [`x ${h.x.toFixed(2)}`, `y ${h.y.toFixed(2)}`];
		if (h.p !== null) parts.push(`P(y=1) ${h.p.toFixed(3)}`);
		if (vote) {
			const v = 2 * vote.ones - vote.k;
			parts.push(`vote ${vote.ones}/${vote.k} → ${v > 0 ? 'class 1' : v < 0 ? 'class 0' : 'tie'}`);
		}
		return parts.join(' · ');
	}

	function toWorld(e: PointerEvent): [number, number] {
		const r = wrap.getBoundingClientRect();
		return [wx(e.clientX - r.left), wy(e.clientY - r.top)];
	}

	function onPointerDown(e: PointerEvent) {
		if (e.button !== 0 && e.button !== 2) return;
		const [x, y] = toWorld(e);
		const other = e.shiftKey || e.button === 2;
		onadd(x, y, other ? ((1 - brush) as 0 | 1) : brush);
	}

	function onKeyDown(e: KeyboardEvent) {
		const st = (D.xMax - D.xMin) * (e.shiftKey ? 0.05 : 0.01);
		let [x, y] = cursor;
		switch (e.key) {
			case 'ArrowLeft':
				x -= st;
				break;
			case 'ArrowRight':
				x += st;
				break;
			case 'ArrowUp':
				y += st;
				break;
			case 'ArrowDown':
				y -= st;
				break;
			case 'Enter':
			case ' ':
				e.preventDefault();
				onadd(x, y, e.shiftKey ? ((1 - brush) as 0 | 1) : brush);
				return;
			default:
				return;
		}
		e.preventDefault();
		cursor = [Math.min(D.xMax, Math.max(D.xMin, x)), Math.min(D.yMax, Math.max(D.yMin, y))];
		hover = { x: cursor[0], y: cursor[1], p: predictor?.predict(cursor[0], cursor[1]) ?? null };
		hoverPt = [cursor[0], cursor[1]];
		drawHover();
	}
</script>

<figure class="boundary">
	<!-- 2D point picker with its own arrow-key handling (see SurfaceCanvas for the rationale) -->
	<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
	<div
		bind:this={wrap}
		class="plot"
		role="application"
		aria-label="Decision boundary plot. Click to add a training point; Shift-click or right-click adds the other class. With focus, arrow keys move the cursor and Enter adds a point."
		tabindex="0"
		onpointerdown={onPointerDown}
		onpointermove={(e) => {
			const [x, y] = toWorld(e);
			hover = { x, y, p: predictor?.predict(x, y) ?? null };
			hoverPt = [x, y];
			drawHover();
		}}
		onpointerleave={() => {
			hover = null;
			hoverPt = null;
			drawHover();
		}}
		oncontextmenu={(e) => e.preventDefault()}
		onkeydown={onKeyDown}
		onfocus={() => (focused = true)}
		onblur={() => (focused = false)}
	>
		<canvas bind:this={fieldCv}></canvas>
		<canvas bind:this={pointCv}></canvas>
		<canvas bind:this={hoverCv}></canvas>
		{#if hover}
			<div class="readout num" aria-hidden="true">
				{readout(hover)}
			</div>
		{/if}
	</div>
</figure>

<style>
	.boundary {
		margin: 0;
	}
	.plot {
		position: relative;
		aspect-ratio: 1;
		width: min(100%, 72vh);
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
