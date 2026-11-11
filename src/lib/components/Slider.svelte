<!--
	Labelled range slider (linear or logarithmic) with a precise numeric entry field.
	One-way `value` + `onchange`, so it works with any state shape.
-->
<script lang="ts">
	interface Props {
		label: string;
		value: number;
		min: number;
		max: number;
		/** Logarithmic mapping (for learning rate, ε, step counts ...). */
		log?: boolean;
		/** Linear step; ignored for log sliders. */
		step?: number;
		integer?: boolean;
		format?: (v: number) => string;
		onchange: (v: number) => void;
		/** Optional KaTeX-free symbol shown before the label (e.g. η). */
		symbol?: string;
	}

	let {
		label,
		value,
		min,
		max,
		log = false,
		step,
		integer = false,
		format = (v) => String(Number(v.toPrecision(3))),
		onchange,
		symbol
	}: Props = $props();

	const uid = $props.id();
	const RES = 1000;

	const pos = $derived(
		log
			? Math.round(((Math.log(value) - Math.log(min)) / (Math.log(max) - Math.log(min))) * RES)
			: value
	);

	function fromPos(p: number): number {
		let v = log ? Math.exp(Math.log(min) + (p / RES) * (Math.log(max) - Math.log(min))) : p;
		if (log) v = Number(v.toPrecision(2));
		if (integer) v = Math.round(v);
		return Math.min(max, Math.max(min, v));
	}

	function onRange(e: Event) {
		onchange(fromPos(Number((e.currentTarget as HTMLInputElement).value)));
	}

	function onNumber(e: Event) {
		const el = e.currentTarget as HTMLInputElement;
		const v = Number(el.value);
		if (!Number.isFinite(v) || el.value.trim() === '') {
			el.value = format(value);
			return;
		}
		const c = Math.min(max, Math.max(min, integer ? Math.round(v) : v));
		el.value = format(c);
		onchange(c);
	}
</script>

<div class="slider">
	<label for="{uid}-r">
		{#if symbol}<span class="sym">{symbol}</span>{/if}
		{label}
	</label>
	<input
		class="num-in num"
		type="text"
		inputmode="decimal"
		aria-label="{label} value"
		value={format(value)}
		onchange={onNumber}
		onkeydown={(e) => e.key === 'Enter' && (e.currentTarget as HTMLInputElement).blur()}
	/>
	<input
		id="{uid}-r"
		type="range"
		min={log ? 0 : min}
		max={log ? RES : max}
		step={log ? 1 : (step ?? (integer ? 1 : (max - min) / 1000))}
		value={pos}
		aria-valuetext={format(value)}
		oninput={onRange}
	/>
</div>

<style>
	.slider {
		display: grid;
		grid-template-columns: 1fr auto;
		align-items: center;
		gap: 2px 8px;
	}
	label {
		font-size: 13px;
		color: var(--text-2);
	}
	.sym {
		font-family: 'KaTeX_Math', 'Times New Roman', serif;
		font-style: italic;
		font-size: 15px;
		color: var(--text);
		margin-right: 4px;
	}
	.num-in {
		width: 8ch;
		text-align: right;
		font-size: 12.5px;
		padding: 3px 6px;
		border: 1px solid transparent;
		border-radius: 5px;
		background: transparent;
	}
	.num-in:hover,
	.num-in:focus {
		border-color: var(--border-strong);
		background: var(--surface);
	}
	input[type='range'] {
		grid-column: 1 / -1;
		width: 100%;
		margin: 2px 0 0;
		accent-color: var(--accent);
		height: 24px;
	}
</style>
