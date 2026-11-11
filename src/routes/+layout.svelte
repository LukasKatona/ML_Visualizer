<script lang="ts">
	import '../app.css';
	import { resolve } from '$app/paths';
	import { page } from '$app/state';
	import { onMount, type Snippet } from 'svelte';
	import AdSlot from '#lib/components/AdSlot.svelte';
	import ThemeToggle from '#lib/components/ThemeToggle.svelte';
	import { SITE } from '#lib/site.config.ts';
	import { theme } from '#lib/theme.svelte.ts';

	let { children }: { children: Snippet } = $props();

	onMount(() => theme.init());

	const isHome = $derived(page.route.id === '/');
</script>

<a class="skip" href="#main">Skip to content</a>

<header class="site-header">
	<div class="inner">
		<a class="brand" href={resolve('/')} aria-current={isHome ? 'page' : undefined}>
			<svg width="22" height="22" viewBox="0 0 32 32" aria-hidden="true">
				<path
					d="M4 25C8 25 10 8 16 8s8 17 12 17"
					fill="none"
					stroke="currentColor"
					stroke-width="2.6"
					stroke-linecap="round"
				/>
				<circle cx="16" cy="8" r="3.4" fill="var(--c-momentum)" />
			</svg>
			<span>{SITE.name}</span>
		</a>
		<nav aria-label="Primary">
			<a href={resolve('/')} class:active={isHome}>Visualizers</a>
		</nav>
		<ThemeToggle />
	</div>
</header>

<main id="main" tabindex="-1">
	{@render children()}
</main>

<footer class="site-footer">
	<AdSlot name="footer" />
	<div class="inner">
		<span>© {new Date().getFullYear()} {SITE.name} · MIT licensed</span>
		<span class="muted">Everything runs in your browser. No cookies, no tracking.</span>
		{#if SITE.repoUrl}
			<!-- external link: no base path to resolve -->
			<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -->
			<a href={SITE.repoUrl} rel="noopener">Source</a>
		{/if}
	</div>
</footer>

<style>
	.skip {
		position: absolute;
		left: 8px;
		top: -48px;
		z-index: 10;
		padding: 8px 12px;
		background: var(--accent);
		color: var(--accent-ink);
		border-radius: var(--radius-sm);
	}
	.skip:focus {
		top: 8px;
	}
	.site-header {
		position: sticky;
		top: 0;
		z-index: 5;
		background: color-mix(in srgb, var(--bg) 85%, transparent);
		backdrop-filter: blur(8px);
		border-bottom: 1px solid var(--border);
	}
	.inner {
		max-width: var(--maxw);
		margin: 0 auto;
		padding: 0 var(--gutter);
		display: flex;
		align-items: center;
		gap: 20px;
	}
	.site-header .inner {
		height: 56px;
	}
	.brand {
		display: inline-flex;
		align-items: center;
		gap: 9px;
		color: var(--text);
		text-decoration: none;
		font-weight: 650;
		letter-spacing: -0.01em;
	}
	nav {
		margin-left: auto;
		display: flex;
		gap: 4px;
	}
	nav a {
		color: var(--text-2);
		text-decoration: none;
		padding: 6px 10px;
		border-radius: var(--radius-sm);
		font-size: 14px;
	}
	nav a:hover,
	nav a.active {
		color: var(--text);
		background: var(--surface-2);
	}
	main {
		min-height: calc(100vh - 56px - 72px);
		outline: none;
	}
	.site-footer {
		border-top: 1px solid var(--border);
		margin-top: 48px;
		padding: 20px 0 28px;
		font-size: 13px;
		color: var(--text-2);
	}
	.site-footer .inner {
		flex-wrap: wrap;
		gap: 8px 20px;
	}
	.muted {
		color: var(--muted);
	}
</style>
