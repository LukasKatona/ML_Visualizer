<script lang="ts">
	import type { VisualizerMeta } from '#lib/site.config.ts';

	let { kind }: { kind: VisualizerMeta['glyph'] } = $props();
</script>

<svg viewBox="0 0 160 96" class="glyph" aria-hidden="true">
	{#if kind === 'descent'}
		{#each [64, 50, 37, 25, 14] as r, i (r)}
			<ellipse
				cx="104"
				cy="54"
				rx={r * 1.25}
				ry={r * 0.62}
				transform="rotate(-18 104 54)"
				class="iso"
				style:opacity={0.35 + i * 0.12}
			/>
		{/each}
		<path d="M14 18 L40 40 L33 52 L61 55 L70 64 L92 57 L104 54" class="traj a" />
		<path d="M14 18 C50 22 70 70 104 54" class="traj b" />
		<circle cx="104" cy="54" r="3.5" class="dot" />
	{:else if kind === 'boundary'}
		<path d="M10 80 C50 70 60 20 150 14" class="traj a" />
		{#each [[30, 30], [52, 22], [40, 48], [70, 14], [22, 54], [62, 38]] as [x, y], i (i)}
			<circle cx={x} cy={y} r="4" class="pt p1" />
		{/each}
		{#each [[100, 70], [124, 52], [86, 82], [134, 76], [110, 40], [140, 34]] as [x, y], i (i)}
			<rect x={x - 3.5} y={y - 3.5} width="7" height="7" rx="1.5" class="pt p2" />
		{/each}
	{:else if kind === 'attention'}
		{#each Array(6) as _, r (r)}
			{#each Array(6) as _, c (c)}
				<rect
					x={34 + c * 16}
					y={2 + r * 15.5}
					width="13"
					height="13"
					rx="2"
					class="cell"
					style:opacity={c <= r ? 0.15 + 0.85 * Math.exp(-((r - c) ** 2) / 3) : 0.06}
				/>
			{/each}
		{/each}
	{:else}
		{#each [0.9, 0.35, 0.62, 0.15, 0.48, 0.78, 0.25] as v, i (i)}
			<rect x={22 + i * 17} y={88 - v * 76} width="11" height={v * 76} rx="2" class="bar" />
		{/each}
		<path d="M14 88 H150" class="axis" />
	{/if}
</svg>

<style>
	.glyph {
		width: 100%;
		height: auto;
		display: block;
	}
	.iso {
		fill: none;
		stroke: var(--border-strong);
		stroke-width: 1;
	}
	.traj {
		fill: none;
		stroke-width: 2;
		stroke-linecap: round;
		stroke-linejoin: round;
	}
	.a {
		stroke: var(--c-sgd);
	}
	.b {
		stroke: var(--c-momentum);
	}
	.dot {
		fill: var(--text);
	}
	.pt {
		stroke: var(--surface);
		stroke-width: 1.5;
	}
	.p1 {
		fill: var(--c-sgd);
	}
	.p2 {
		fill: var(--c-momentum);
	}
	.cell {
		fill: var(--accent);
	}
	.bar {
		fill: var(--muted);
	}
	.axis {
		stroke: var(--border-strong);
	}
</style>
