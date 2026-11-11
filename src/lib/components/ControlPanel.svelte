<!-- Titled panel used for every control group across visualizers. -->
<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		title: string;
		children: Snippet;
		/** Optional controls rendered at the right of the header. */
		actions?: Snippet;
	}

	let { title, children, actions }: Props = $props();
	const uid = $props.id();
</script>

<section class="panel" aria-labelledby={uid}>
	<header>
		<h2 id={uid}>{title}</h2>
		{#if actions}<div class="actions">{@render actions()}</div>{/if}
	</header>
	<div class="content">{@render children()}</div>
</section>

<style>
	.panel {
		min-width: 0;
		background: var(--surface);
		border: 1px solid var(--border);
		border-radius: var(--radius);
	}
	header {
		display: flex;
		align-items: center;
		gap: 8px;
		min-height: 42px;
		padding: 6px 14px;
		border-bottom: 1px solid var(--border);
	}
	h2 {
		font-family: var(--font-mono);
		font-size: 11px;
		font-weight: 600;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--muted);
	}
	.actions {
		margin-left: auto;
		display: flex;
		gap: 6px;
	}
	.content {
		padding: 12px 14px 14px;
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 12px;
	}
</style>
