<script lang="ts">
	import { theme } from '#lib/theme.svelte.ts';

	const next = $derived(theme.current === 'dark' ? 'light' : 'dark');
</script>

<button
	type="button"
	class="btn icon"
	onclick={() => theme.toggle()}
	aria-label={`Switch to ${next} theme`}
	title={`Switch to ${next} theme`}
>
	{#if theme.current === 'dark'}
		<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
			<circle cx="12" cy="12" r="4.5" fill="currentColor" />
			<g stroke="currentColor" stroke-width="2" stroke-linecap="round">
				<path
					d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"
				/>
			</g>
		</svg>
	{:else}
		<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
			<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5Z" fill="currentColor" />
		</svg>
	{/if}
</button>
