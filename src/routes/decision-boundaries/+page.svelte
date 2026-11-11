<script lang="ts">
	import Seo from '#lib/components/Seo.svelte';
	import { visualizer } from '#lib/site.config.ts';
	import DecisionBoundaryViz from '#lib/viz/decision-boundaries/DecisionBoundaryViz.svelte';

	let { data } = $props();

	const meta = visualizer('decision-boundaries');
</script>

<Seo
	title={meta.title}
	description="Interactive decision boundary visualizer: train logistic regression and neural networks live, compare with k-NN and decision trees on moons, XOR, circles and spirals."
	path="/decision-boundaries/"
/>

<div class="page">
	<header class="head">
		<p class="eyebrow">Visualizer 02 · Classification</p>
		<h1>{meta.title}</h1>
	</header>
	<DecisionBoundaryViz tex={data.tex} />
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
