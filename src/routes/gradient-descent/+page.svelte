<script lang="ts">
	import Seo from '#lib/components/Seo.svelte';
	import { visualizer } from '#lib/site.config.ts';
	import GradientDescentViz from '#lib/viz/gradient-descent/GradientDescentViz.svelte';

	let { data } = $props();

	const meta = visualizer('gradient-descent');
</script>

<Seo
	title={meta.title}
	description="Interactive gradient descent visualizer: race SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam and AdamW on Rosenbrock, Himmelblau, Beale, saddle and noisy loss surfaces."
	path="/gradient-descent/"
/>

<div class="page">
	<header class="head">
		<p class="eyebrow">Visualizer 01 · Optimization</p>
		<h1>{meta.title}</h1>
	</header>
	<GradientDescentViz tex={data.tex} />
</div>

<style>
	.page {
		max-width: var(--maxw);
		margin: 0 auto;
		padding: 24px var(--gutter) 0;
	}
	.head {
		margin-bottom: 18px;
	}
	h1 {
		font-size: clamp(22px, 3vw, 30px);
		font-weight: 660;
		margin-top: 6px;
	}
</style>
