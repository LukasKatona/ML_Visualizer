<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { onMount } from 'svelte';
	import 'katex/dist/katex.min.css';
	import ControlPanel from '#lib/components/ControlPanel.svelte';
	import LineChart from '#lib/components/LineChart.svelte';
	import Slider from '#lib/components/Slider.svelte';
	import { DATASETS, DATASET_IDS, generate, withExtra, type DatasetId } from '#lib/ml/datasets.ts';
	import { MODEL_DOCS } from '#lib/ml/docs.ts';
	import { VEC_DEFAULTS, type VecOptimizerId } from '#lib/ml/optim.ts';
	import {
		DB_LIMITS,
		MODELS,
		MODEL_IDS,
		Session,
		decodeDb,
		defaultDbState,
		encodeDb,
		type DbState,
		type ModelId
	} from '#lib/ml/session.ts';
	import BoundaryCanvas from './BoundaryCanvas.svelte';

	interface Props {
		tex: Record<ModelId, string>;
	}

	let { tex }: Props = $props();

	let st = $state<DbState>(defaultDbState());
	let playing = $state(false);
	let showTest = $state(true);
	let brush = $state<0 | 1>(0);
	let version = $state(0);
	let canvas: BoundaryCanvas | undefined = $state();
	let mounted = false;
	let announce = $state('');

	// data depends only on dataset parameters and user points
	const data = $derived.by(() => {
		const { train, test } = generate(st.dataset, st.n, st.noise, st.seed);
		return { train: withExtra(train, st.extra), test };
	});

	// Any model / data change rebuilds the session (training restarts from the seed init).
	// Derived (not an effect) so it also exists during prerendering: no layout shift on load.
	const session = $derived(new Session($state.snapshot(st), data.train, data.test));

	const trainable = $derived(MODELS[st.model].trainable);

	function play() {
		if (!session.trainable) return;
		if (session.status === 'done' || session.status === 'diverged') reset();
		playing = true;
	}
	function pause() {
		playing = false;
		canvas?.paint();
		version++;
	}
	function step() {
		if (!session.trainable) return;
		playing = false;
		session.advance(1);
		canvas?.paint();
		version++;
	}
	function reset() {
		playing = false;
		session.reset();
		canvas?.paint();
		version++;
	}

	onMount(() => {
		if (window.location.search) st = decodeDb(window.location.search);
		mounted = true;
		let raf = 0;
		let frames = 0;
		const loop = () => {
			raf = requestAnimationFrame(loop);
			if (!playing) return;
			const active = session.advance(st.speed);
			frames++;
			if (!active) {
				playing = false;
				canvas?.paint();
				version++;
				return;
			}
			if (frames % 3 === 0) canvas?.paint(true);
			if (frames % 6 === 0) version++;
		};
		raf = requestAnimationFrame(loop);
		return () => cancelAnimationFrame(raf);
	});

	$effect(() => {
		const q = encodeDb(st);
		if (!mounted) return;
		const id = setTimeout(() => {
			if (window.location.search.slice(1) !== q)
				goto(resolve(`decision-boundaries/?${q}`), { shallow: true, replace: true });
		}, 300);
		return () => clearTimeout(id);
	});

	function addPoint(x: number, y: number, cls: 0 | 1) {
		if (st.extra.length >= DB_LIMITS.extra) {
			announce = `At most ${DB_LIMITS.extra} added points`;
			return;
		}
		st.extra = [...st.extra, [x, y, cls]];
	}

	function setOptimizer(o: VecOptimizerId) {
		st.optimizer = o;
		st.lr = VEC_DEFAULTS[o].lr;
	}

	async function share() {
		try {
			await navigator.clipboard.writeText(window.location.href);
			announce = 'Link copied to clipboard';
		} catch {
			announce = 'Copy the URL from the address bar to share this setup';
		}
		setTimeout(() => (announce = ''), 2500);
	}

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

	const metrics = $derived.by(() => {
		void version;
		const s = session;
		const acc = s.accuracies();
		return {
			acc,
			epoch: s.epoch,
			loss: s.mlp ? s.loss[s.count - 1] : null,
			status: s.status,
			info: s.info,
			count: s.count
		};
	});

	const pct = (v: number) => `${(100 * v).toFixed(1)}%`;
</script>

<svelte:window onkeydown={onKey} />

<div class="viz">
	<div class="plot-area">
		<BoundaryCanvas
			bind:this={canvas}
			predictor={session.predictor}
			train={data.train}
			test={data.test}
			{showTest}
			{brush}
			onadd={addPoint}
		/>

		<div class="transport" role="toolbar" aria-label="Training">
			{#if trainable}
				{#if playing}
					<button class="btn primary" onclick={pause} aria-keyshortcuts="Space K">Pause</button>
				{:else}
					<button class="btn primary" onclick={play} aria-keyshortcuts="Space K">
						{metrics && metrics.epoch > 0 && metrics.status === 'training' ? 'Resume' : 'Train'}
					</button>
				{/if}
				<button class="btn" onclick={step} aria-keyshortcuts="Period L">Step</button>
				<button class="btn" onclick={reset} aria-keyshortcuts="R">Re-init</button>
			{/if}
			<button class="btn share" onclick={share}>Share link</button>
			<span class="visually-hidden" aria-live="polite">{announce}</span>
			{#if announce}<span class="toast" aria-hidden="true">{announce}</span>{/if}
		</div>

		{#if metrics}
			<dl class="stats">
				{#if trainable}
					<div>
						<dt>Epoch</dt>
						<dd class="num">{metrics.epoch}</dd>
					</div>
					<div>
						<dt>Train loss</dt>
						<dd class="num">{metrics.loss?.toFixed(4)}</dd>
					</div>
				{/if}
				<div>
					<dt>Train acc.</dt>
					<dd class="num">{pct(metrics.acc.train)}</dd>
				</div>
				<div>
					<dt>Test acc.</dt>
					<dd class="num">{pct(metrics.acc.test)}</dd>
				</div>
				<div>
					<dt>Model</dt>
					<dd>{metrics.info}</dd>
				</div>
				{#if metrics.status === 'diverged'}
					<div>
						<dt>Status</dt>
						<dd class="bad">diverged; lower η</dd>
					</div>
				{/if}
			</dl>
		{/if}

		<div class="legend-row">
			<span><span class="mk c0" aria-hidden="true"></span>Class 0</span>
			<span><span class="mk c1" aria-hidden="true"></span>Class 1</span>
			<span><span class="mk c0 hollow" aria-hidden="true"></span>Test point (not trained on)</span>
			<span><span class="ln" aria-hidden="true"></span>P = 0.5 boundary</span>
		</div>
		<p class="hint">
			Click to add a <strong>class {brush}</strong> point · Shift-click or right-click adds class {1 -
				brush}
			{#if trainable}· <kbd>Space</kbd> train/pause · <kbd>.</kbd> step · <kbd>R</kbd> re-init{/if}
		</p>

		{#if trainable}
			<div class="charts">
				<ControlPanel title="Training loss">
					<LineChart
						series={[
							{ id: 'loss', label: 'Train loss', color: 'var(--series-1)', data: session.loss }
						]}
						count={metrics?.count ?? 0}
						{version}
						logY
						label="Training loss (binary cross-entropy + L2) per epoch, log scale"
					/>
				</ControlPanel>
				<ControlPanel title="Accuracy">
					<LineChart
						series={[
							{
								id: 'train',
								label: 'Train',
								color: 'var(--series-1)',
								data: session.trainAcc
							},
							{
								id: 'test',
								label: 'Test',
								color: 'var(--series-2)',
								data: session.testAcc,
								dashed: true
							}
						]}
						count={metrics?.count ?? 0}
						{version}
						yDomain={[0.4, 1]}
						yFormat={(v) => `${Math.round(v * 100)}%`}
						label="Train and test accuracy per epoch"
					/>
				</ControlPanel>
			</div>
		{/if}
	</div>

	<aside class="side" aria-label="Controls">
		<ControlPanel title="Data">
			<label class="field">
				<span class="visually-hidden">Dataset</span>
				<select
					value={st.dataset}
					onchange={(e) => {
						const id = e.currentTarget.value as DatasetId;
						st.dataset = id;
						st.noise = DATASETS[id].noise;
						st.extra = [];
					}}
				>
					{#each DATASET_IDS as id (id)}
						<option value={id}>{DATASETS[id].name}</option>
					{/each}
				</select>
			</label>
			<p class="note">{DATASETS[st.dataset].note}</p>
			<Slider
				label="Points"
				value={st.n}
				min={DB_LIMITS.n.min}
				max={DB_LIMITS.n.max}
				integer
				step={10}
				format={(v) => String(v)}
				onchange={(v) => (st.n = v)}
			/>
			<Slider
				label="Noise σ"
				value={st.noise}
				min={DB_LIMITS.noise.min}
				max={DB_LIMITS.noise.max}
				step={0.01}
				format={(v) => v.toFixed(2)}
				onchange={(v) => (st.noise = v)}
			/>
			<div class="row">
				<div class="seg" role="radiogroup" aria-label="Class placed by click">
					{#each [0, 1] as const as c (c)}
						<button
							role="radio"
							aria-checked={brush === c}
							class:on={brush === c}
							onclick={() => (brush = c)}
						>
							<span class="mk c{c}" aria-hidden="true"></span>Class {c}
						</button>
					{/each}
				</div>
				<button class="btn small" onclick={() => (st.seed = (st.seed % 9999) + 1)}>Resample</button>
			</div>
			<div class="row">
				<label class="check">
					<input type="checkbox" bind:checked={showTest} />
					Show test points
				</label>
				{#if st.extra.length}
					<button class="btn small" onclick={() => (st.extra = [])}>
						Clear {st.extra.length} added
					</button>
				{/if}
			</div>
		</ControlPanel>

		<ControlPanel title="Model">
			<label class="field">
				<span class="visually-hidden">Model</span>
				<select value={st.model} onchange={(e) => (st.model = e.currentTarget.value as ModelId)}>
					{#each MODEL_IDS as id (id)}
						<option value={id}>{MODELS[id].name}</option>
					{/each}
				</select>
			</label>
			{#if st.model === 'logreg' || st.model === 'mlp'}
				<Slider
					label="Polynomial feature degree d"
					value={st.degree}
					min={DB_LIMITS.degree.min}
					max={DB_LIMITS.degree.max}
					integer
					step={1}
					format={(v) => String(v)}
					onchange={(v) => (st.degree = v)}
				/>
			{/if}
			{#if st.model === 'mlp'}
				<Slider
					label="Hidden layers"
					value={st.layers}
					min={DB_LIMITS.layers.min}
					max={DB_LIMITS.layers.max}
					integer
					step={1}
					format={(v) => String(v)}
					onchange={(v) => (st.layers = v)}
				/>
				<Slider
					label="Width (units per layer)"
					value={st.width}
					min={DB_LIMITS.width.min}
					max={DB_LIMITS.width.max}
					integer
					step={1}
					format={(v) => String(v)}
					onchange={(v) => (st.width = v)}
				/>
				<div class="seg" role="radiogroup" aria-label="Activation">
					{#each ['tanh', 'relu'] as const as a (a)}
						<button
							role="radio"
							aria-checked={st.activation === a}
							class:on={st.activation === a}
							onclick={() => (st.activation = a)}>{a === 'tanh' ? 'tanh' : 'ReLU'}</button
						>
					{/each}
				</div>
			{/if}
			{#if st.model === 'knn'}
				<Slider
					label="Neighbours k"
					value={st.k}
					min={DB_LIMITS.k.min}
					max={DB_LIMITS.k.max}
					integer
					step={1}
					format={(v) => String(v)}
					onchange={(v) => (st.k = v)}
				/>
			{/if}
			{#if st.model === 'tree'}
				<Slider
					label="Max depth"
					value={st.maxDepth}
					min={DB_LIMITS.maxDepth.min}
					max={DB_LIMITS.maxDepth.max}
					integer
					step={1}
					format={(v) => String(v)}
					onchange={(v) => (st.maxDepth = v)}
				/>
			{/if}
		</ControlPanel>

		{#if trainable}
			<ControlPanel title="Training">
				<div class="seg" role="radiogroup" aria-label="Optimizer">
					{#each ['sgd', 'momentum', 'adam'] as const as o (o)}
						<button
							role="radio"
							aria-checked={st.optimizer === o}
							class:on={st.optimizer === o}
							onclick={() => setOptimizer(o)}
							>{o === 'sgd' ? 'SGD' : o === 'momentum' ? 'Momentum' : 'Adam'}</button
						>
					{/each}
				</div>
				<Slider
					label="Learning rate η"
					value={st.lr}
					min={DB_LIMITS.lr.min}
					max={DB_LIMITS.lr.max}
					log
					format={(v) => String(Number(v.toPrecision(3)))}
					onchange={(v) => (st.lr = v)}
				/>
				<Slider
					label="L2 penalty λ"
					value={st.l2}
					min={DB_LIMITS.l2.min}
					max={DB_LIMITS.l2.max}
					step={0.0005}
					format={(v) => String(Number(v.toPrecision(3)))}
					onchange={(v) => (st.l2 = v)}
				/>
				<Slider
					label="Speed (epochs / frame)"
					value={st.speed}
					min={DB_LIMITS.speed.min}
					max={DB_LIMITS.speed.max}
					log
					integer
					format={(v) => String(v)}
					onchange={(v) => (st.speed = v)}
				/>
				<Slider
					label="Max epochs"
					value={st.maxEpochs}
					min={DB_LIMITS.maxEpochs.min}
					max={DB_LIMITS.maxEpochs.max}
					log
					integer
					format={(v) => String(v)}
					onchange={(v) => (st.maxEpochs = v)}
				/>
				<p class="note small">
					Full-batch gradient descent on the training points. Seed {st.seed} fixes data and initialisation.
				</p>
			</ControlPanel>
		{/if}

		<ControlPanel title={MODELS[st.model].name}>
			<!-- trusted: KaTeX output of constant formulas, rendered at build time -->
			<!-- eslint-disable-next-line svelte/no-at-html-tags -->
			<div class="tex">{@html tex[st.model]}</div>
			<p class="note"><strong>Works well:</strong> {MODEL_DOCS[st.model].good}</p>
			<p class="note"><strong>Struggles:</strong> {MODEL_DOCS[st.model].bad}</p>
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
		grid-template-columns: minmax(0, 1fr);
		gap: 12px;
	}
	.side {
		grid-area: side;
		min-width: 0;
		display: grid;
		grid-template-columns: minmax(0, 1fr);
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
	.stats {
		display: flex;
		flex-wrap: wrap;
		gap: 8px 22px;
		margin: 0;
	}
	.stats div {
		display: grid;
	}
	.stats dt {
		font-size: 11.5px;
		color: var(--muted);
	}
	.stats dd {
		margin: 0;
		font-size: 15px;
	}
	.bad {
		color: var(--danger);
	}
	.legend-row {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 16px;
		font-size: 12.5px;
		color: var(--text-2);
	}
	.legend-row > span {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
	.mk {
		display: inline-block;
		width: 10px;
		height: 10px;
		flex: none;
	}
	.mk.c0 {
		border-radius: 50%;
		background: var(--class-0);
	}
	.mk.c1 {
		border-radius: 1.5px;
		background: var(--class-1);
	}
	.mk.hollow {
		background: transparent;
		border: 2px solid var(--class-0);
	}
	.ln {
		display: inline-block;
		width: 18px;
		height: 0;
		border-top: 2px solid var(--text);
	}
	.hint {
		margin: 0;
		font-size: 12.5px;
		color: var(--muted);
	}
	.hint strong {
		color: var(--text-2);
		font-weight: 600;
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
	.charts {
		display: grid;
		gap: 12px;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr));
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
	.note.small {
		font-size: 12px;
		color: var(--muted);
	}
	.note strong {
		color: var(--text);
		font-weight: 600;
	}
	.row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 8px;
		flex-wrap: wrap;
	}
	.seg {
		display: inline-flex;
		border: 1px solid var(--border-strong);
		border-radius: var(--radius-sm);
		overflow: hidden;
	}
	.seg button {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		min-height: 32px;
		padding: 0 11px;
		border: 0;
		background: var(--surface);
		font-size: 13px;
		cursor: pointer;
	}
	.seg button + button {
		border-left: 1px solid var(--border-strong);
	}
	.seg button.on {
		background: var(--surface-2);
		font-weight: 600;
		box-shadow: inset 0 -2px 0 var(--accent);
	}
	.check {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font-size: 13px;
		color: var(--text-2);
		cursor: pointer;
	}
	.check input {
		accent-color: var(--accent);
		width: 15px;
		height: 15px;
		margin: 0;
	}
	.btn.small {
		min-height: 30px;
		padding: 0 10px;
		font-size: 12.5px;
	}
	.tex {
		overflow-x: auto;
		overflow-y: hidden;
		font-size: 13.5px;
		scrollbar-width: thin;
	}
	.tex :global(.katex-display) {
		margin: 0.2em 0;
	}
</style>
