<script lang="ts">
	import { FEATURES } from '#lib/site.config.ts';

	interface Props {
		/** Identifier for the ad network integration, e.g. 'footer' or 'sidebar'. */
		name: string;
		variant?: 'banner' | 'rail';
	}

	let { name, variant = 'banner' }: Props = $props();
</script>

{#if FEATURES.ads}
	<aside class="ad-slot {variant}" data-ad-slot={name} aria-label="Advertisement"></aside>
{/if}

<style>
	.ad-slot {
		border: 1px dashed var(--border);
		border-radius: var(--radius-sm);
		margin-inline: auto;
	}
	.banner {
		width: 100%;
		max-width: 728px;
		min-height: 90px;
	}
	.rail {
		width: 100%;
		min-height: 250px;
	}
</style>
