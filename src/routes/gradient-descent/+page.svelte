<script lang="ts">
	import Seo from '#lib/components/Seo.svelte';
	import { visualizer } from '#lib/site.config.ts';

	const meta = visualizer('gradient-descent');
</script>

<Seo title={meta.title} description={meta.description} path="/gradient-descent/" />

<h1>{meta.title}</h1>
