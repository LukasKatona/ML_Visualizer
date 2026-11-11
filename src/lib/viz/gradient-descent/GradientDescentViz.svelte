<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { onMount, untrack } from 'svelte';
	import ControlPanel from '#lib/components/ControlPanel.svelte';
	import Slider from '#lib/components/Slider.svelte';
	import { SURFACES, SURFACE_IDS, type SurfaceId } from '#lib/math/surfaces.ts';
	import {
		HYPER_META,
		OPTIMIZERS,
		OPTIMIZER_IDS,
		defaultHyper,
		type Hyper,
		type HyperKey,
		type OptimizerId
	} from '#lib/optimizers/optimizers.ts';
	import { Simulation, type RunStatus } from '#lib/sim/engine.ts';
	import { LIMITS, decode, defaultState, encode, fmt, type VizState } from '#lib/sim/urlState.ts';
	import SurfaceCanvas from './SurfaceCanvas.svelte';

	// ---------- state (mirrors VizState; hyperparameters kept for all optimizers) ----------
	const init = defaultState();
	let surfaceId = $state<SurfaceId>(init.surface);
	let start = $state<[number, number]>(init.start);
	let enabled = $state<OptimizerId[]>(init.runs.map((r) => r.id));
	let hypers = $state<Record<OptimizerId, Hyper>>(allHypers(init.surface));
	let selected = $state<OptimizerId>(init.selected);
	let maxSteps = $state(init.maxSteps);
	let speed = $state(init.speed);
	let noise = $state(init.noise);
	let seed = $state(init.seed);
	let logY = $state(init.logY);

	let playing = $state(false);
	let sim = $state.raw<Simulation | null>(null);
	/** Bumped (throttled) while running so the stats table refreshes without per-frame renders. */
	let version = $state(0);
	let mounted = false;
	let announce = $state('');

	let canvas: SurfaceCanvas | undefined = $state();

	const surface = $derived(SURFACES[surfaceId]);

	function allHypers(s: SurfaceId): Record<OptimizerId, Hyper> {
		const lr = SURFACES[s].lr;
		return Object.fromEntries(OPTIMIZER_IDS.map((id) => [id, defaultHyper(id, lr)])) as Record<
			OptimizerId,
			Hyper
		>;
	}

	function toVizState(): VizState {
		return {
			surface: surfaceId,
			start: [start[0], start[1]],
			runs: enabled.map((id) => ({ id, h: { ...hypers[id] } })),
			selected,
			maxSteps,
			speed,
			noise,
			seed,
			logY
		};
	}

	function apply(st: VizState) {
		surfaceId = st.surface;
		start = st.start;
		const h = allHypers(st.surface);
		for (const r of st.runs) h[r.id] = r.h;
		hypers = h;
		enabled = OPTIMIZER_IDS.filter((id) => st.runs.some((r) => r.id === id));
		selected = st.selected;
		maxSteps = st.maxSteps;
		speed = st.speed;
		noise = st.noise;
		seed = st.seed;
		logY = st.logY;
	}

	function changeSurface(id: SurfaceId) {
		const d = defaultState(id);
		playing = false;
		surfaceId = id;
		start = d.start;
		hypers = allHypers(id);
		noise = d.noise;
	}

	// ---------- simulation lifecycle ----------
	// Any structural change rebuilds the race from the start point (keeps play state).
	$effect(() => {
		const cfg = enabled.map((id) => ({ id, h: { ...hypers[id] } }));
		const s = new Simulation(SURFACES[surfaceId], [start[0], start[1]], cfg, maxSteps, noise, seed);
		sim = s;
		untrack(() => version++);
	});

	function play() {
		if (!sim) return;
		if (sim.finished) reset();
		playing = true;
	}
	function pause() {
		playing = false;
	}
	function step() {
		if (!sim) return;
		playing = false;
		sim.advance(1);
		canvas?.frame();
		version++;
	}
	function reset() {
		playing = false;
		sim?.reset();
		canvas?.redraw();
		version++;
	}

	onMount(() => {
		if (window.location.search) apply(decode(window.location.search));
		mounted = true;

		let raf = 0;
		let frames = 0;
		const loop = () => {
			raf = requestAnimationFrame(loop);
			if (!playing || !sim) return;
			const active = sim.advance(speed);
			canvas?.frame();
			if (!active) {
				playing = false;
				version++;
			} else if (++frames % 6 === 0) version++;
		};
		raf = requestAnimationFrame(loop);
		return () => cancelAnimationFrame(raf);
	});

	// ---------- URL sync (debounced, shallow: no navigation) ----------
	$effect(() => {
		const q = encode(toVizState());
		if (!mounted) return;
		const id = setTimeout(() => {
			if (window.location.search.slice(1) !== q)
				goto(resolve(`gradient-descent/?${q}`), { shallow: true, replace: true });
		}, 300);
		return () => clearTimeout(id);
	});

	async function share() {
		try {
			await navigator.clipboard.writeText(window.location.href);
			announce = 'Link copied to clipboard';
		} catch {
			announce = 'Copy the URL from the address bar to share this setup';
		}
		setTimeout(() => (announce = ''), 2500);
	}

	// ---------- keyboard shortcuts ----------
	function onKey(e: KeyboardEvent) {
		const t = e.target as HTMLElement;
		if (e.ctrlKey || e.metaKey || e.altKey) return;
		if (t.closest('input, select, textarea, [role="application"]')) return;
		if (e.key === ' ' || e.key === 'k') {
			if (t.tagName === 'BUTTON' && e.key === ' ') return;
			e.preventDefault();
			if (playing) pause();
			else play();
		} else if (e.key === '.' || e.key === 'l') step();
		else if (e.key === 'r') reset();
	}

	// ---------- helpers for the template ----------
	function toggle(id: OptimizerId, on: boolean) {
		enabled = OPTIMIZER_IDS.filter((x) => (x === id ? on : enabled.includes(x)));
		if (on) selected = id;
	}

	function setHyper(key: HyperKey, v: number) {
		hypers[selected] = { ...hypers[selected], [key]: v };
	}

	function resetHypers() {
		hypers[selected] = defaultHyper(selected, surface.lr);
	}

	const STATUS_LABEL: Record<RunStatus, string> = {
		running: 'running',
		converged: 'converged',
		diverged: 'diverged',
		escaped: 'left domain',
		done: 'max steps'
	};

	const rows = $derived.by(() => {
		void version;
		return OPTIMIZER_IDS.map((id) => {
			const run = sim?.runs.find((r) => r.id === id) ?? null;
			return {
				id,
				name: OPTIMIZERS[id].name,
				on: enabled.includes(id),
				steps: run ? run.steps : null,
				gap: run ? run.lastLoss : null,
				status: run ? run.status : null
			};
		});
	});

	const stepsDone = $derived.by(() => {
		void version;
		return sim?.t ?? 0;
	});
</script>

<svelte:window onkeydown={onKey} />

<div class="viz">
	<div class="plot-area">
		<SurfaceCanvas
			bind:this={canvas}
			{surface}
			{sim}
			{start}
			{selected}
			onpick={(x, y) => {
				start = [x, y];
			}}
		/>
		<div class="transport" role="toolbar" aria-label="Playback">
			{#if playing}
				<button class="btn primary" onclick={pause} aria-keyshortcuts="Space K">
					<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"
						><rect x="2" y="1" width="3.5" height="12" rx="1" fill="currentColor" /><rect
							x="8.5"
							y="1"
							width="3.5"
							height="12"
							rx="1"
							fill="currentColor"
						/></svg
					>
					Pause
				</button>
			{:else}
				<button
					class="btn primary"
					onclick={play}
					aria-keyshortcuts="Space K"
					disabled={!enabled.length}
				>
					<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"
						><path d="M3 1.5v11l9.5-5.5z" fill="currentColor" /></svg
					>
					{stepsDone > 0 && !sim?.finished ? 'Resume' : 'Play'}
				</button>
			{/if}
			<button class="btn" onclick={step} aria-keyshortcuts="Period L" disabled={!enabled.length}>
				Step
			</button>
			<button class="btn" onclick={reset} aria-keyshortcuts="R">Reset</button>
			<span class="counter num" aria-live="off">t = {stepsDone} / {maxSteps}</span>
			<button class="btn share" onclick={share}>Share link</button>
			<span class="visually-hidden" aria-live="polite">{announce}</span>
			{#if announce}<span class="toast" aria-hidden="true">{announce}</span>{/if}
		</div>
		<p class="hint">
			Click the surface to set the start point · <kbd>Space</kbd> play/pause · <kbd>.</kbd> step ·
			<kbd>R</kbd> reset
		</p>
	</div>

	<aside class="side" aria-label="Controls">
		<ControlPanel title="Loss surface">
			<label class="field">
				<span class="visually-hidden">Surface</span>
				<select
					value={surfaceId}
					onchange={(e) => changeSurface(e.currentTarget.value as SurfaceId)}
				>
					{#each SURFACE_IDS as id (id)}
						<option value={id}>{SURFACES[id].name}</option>
					{/each}
				</select>
			</label>
			<p class="note">{surface.note}</p>
		</ControlPanel>

		<ControlPanel title="Optimizers">
			<table class="runs">
				<caption class="visually-hidden">
					Optimizers in the race, with steps taken, loss gap f − f* and status
				</caption>
				<thead>
					<tr>
						<th scope="col"><span class="visually-hidden">In race</span></th>
						<th scope="col">Optimizer</th>
						<th scope="col" class="r">Steps</th>
						<th scope="col" class="r">f − f*</th>
						<th scope="col" class="r">Status</th>
					</tr>
				</thead>
				<tbody>
					{#each rows as r (r.id)}
						<tr class:sel={r.id === selected} class:off={!r.on}>
							<td>
								<input
									type="checkbox"
									checked={r.on}
									aria-label={`Include ${r.name} in the race`}
									onchange={(e) => toggle(r.id, e.currentTarget.checked)}
								/>
							</td>
							<td>
								<button
									class="opt"
									aria-pressed={r.id === selected}
									onclick={() => (selected = r.id)}
									title="Edit hyperparameters and show the update rule"
								>
									<span class="sw" style:background={`var(--c-${r.id})`}></span>
									{r.name}
								</button>
							</td>
							<td class="r num">{r.steps ?? '–'}</td>
							<td class="r num">{r.gap === null ? '–' : r.gap.toExponential(1)}</td>
							<td class="r status" data-s={r.status}>{r.status ? STATUS_LABEL[r.status] : ''}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</ControlPanel>

		<ControlPanel title={`${OPTIMIZERS[selected].name} hyperparameters`}>
			{#snippet actions()}
				<button class="btn small" onclick={resetHypers}>Defaults</button>
			{/snippet}
			{#each OPTIMIZERS[selected].hypers as key (selected + key)}
				{@const m = HYPER_META[key]}
				<Slider
					label={m.label}
					value={hypers[selected][key] ?? 0}
					min={m.min}
					max={m.max}
					log={m.log}
					step={m.log ? undefined : 0.001}
					format={fmt}
					onchange={(v) => setHyper(key, v)}
				/>
			{/each}
			{#if !enabled.includes(selected)}
				<p class="note">Not in the race. Tick its checkbox to run it.</p>
			{/if}
		</ControlPanel>

		<ControlPanel title="Simulation">
			<Slider
				label="Speed (steps / frame)"
				value={speed}
				min={LIMITS.speed.min}
				max={LIMITS.speed.max}
				log
				integer
				format={(v) => String(v)}
				onchange={(v) => (speed = v)}
			/>
			<Slider
				label="Max steps"
				value={maxSteps}
				min={LIMITS.maxSteps.min}
				max={LIMITS.maxSteps.max}
				log
				integer
				format={(v) => String(v)}
				onchange={(v) => (maxSteps = v)}
			/>
			<Slider
				label="Gradient noise σ"
				value={noise}
				min={LIMITS.noise.min}
				max={LIMITS.noise.max}
				step={0.05}
				format={fmt}
				onchange={(v) => (noise = v)}
			/>
			{#if noise > 0}
				<div class="seed">
					<span>Noise seed <span class="num">{seed}</span></span>
					<button class="btn small" onclick={() => (seed = (seed % 9999) + 1)}>New seed</button>
				</div>
			{/if}
		</ControlPanel>
	</aside>
</div>

<style>
	.viz {
		display: grid;
		gap: 20px;
		grid-template-columns: minmax(0, 1fr);
		grid-template-areas: 'plot' 'side';
	}
	@media (min-width: 980px) {
		.viz {
			grid-template-columns: minmax(0, 1fr) 360px;
			grid-template-areas: 'plot side';
			align-items: start;
		}
	}
	.plot-area {
		grid-area: plot;
		min-width: 0;
		display: grid;
		gap: 12px;
	}
	.side {
		grid-area: side;
		display: grid;
		gap: 12px;
		align-content: start;
	}
	.transport {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 8px;
		position: relative;
	}
	.counter {
		font-size: 13px;
		color: var(--text-2);
		margin-left: 4px;
	}
	.share {
		margin-left: auto;
	}
	.toast {
		position: absolute;
		right: 0;
		top: calc(100% + 6px);
		font-size: 12.5px;
		background: var(--text);
		color: var(--bg);
		padding: 4px 10px;
		border-radius: 6px;
		z-index: 2;
	}
	.hint {
		margin: 0;
		font-size: 12.5px;
		color: var(--muted);
	}
	kbd {
		font-family: var(--font-mono);
		font-size: 11px;
		padding: 1px 5px;
		border: 1px solid var(--border-strong);
		border-bottom-width: 2px;
		border-radius: 4px;
		background: var(--surface);
	}
	select {
		width: 100%;
		min-height: 36px;
		padding: 0 10px;
		border: 1px solid var(--border-strong);
		border-radius: var(--radius-sm);
		background: var(--surface);
	}
	.note {
		margin: 0;
		font-size: 13px;
		color: var(--text-2);
	}
	.runs {
		width: 100%;
		border-collapse: collapse;
		font-size: 13px;
	}
	.runs th {
		font-weight: 500;
		font-size: 11.5px;
		color: var(--muted);
		text-align: left;
		padding: 0 4px 6px;
	}
	.runs td {
		padding: 2px 4px;
		border-top: 1px solid var(--border);
	}
	.runs .r {
		text-align: right;
	}
	.runs tr.off td:not(:first-child) {
		opacity: 0.55;
	}
	.runs input[type='checkbox'] {
		width: 16px;
		height: 16px;
		accent-color: var(--accent);
		margin: 0;
		vertical-align: middle;
	}
	.opt {
		display: inline-flex;
		align-items: center;
		gap: 7px;
		min-height: 30px;
		padding: 0 6px;
		margin-left: -6px;
		border: 0;
		border-radius: 5px;
		background: transparent;
		cursor: pointer;
		font-weight: 500;
	}
	.opt:hover {
		background: var(--surface-2);
	}
	.opt[aria-pressed='true'] {
		background: var(--surface-2);
		box-shadow: inset 0 0 0 1px var(--border-strong);
	}
	.sw {
		width: 12px;
		height: 12px;
		border-radius: 3px;
		flex: none;
	}
	.status {
		font-size: 11.5px;
		color: var(--muted);
		white-space: nowrap;
	}
	.status[data-s='diverged'] {
		color: var(--danger);
	}
	.btn.small {
		min-height: 28px;
		padding: 0 9px;
		font-size: 12.5px;
	}
	.seed {
		display: flex;
		align-items: center;
		justify-content: space-between;
		font-size: 13px;
		color: var(--text-2);
	}
</style>
