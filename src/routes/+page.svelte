<script lang="ts">
	import { resolve } from '$app/paths';
	import Glyph from '#lib/components/Glyph.svelte';
	import Seo from '#lib/components/Seo.svelte';
	import { SITE, VISUALIZERS } from '#lib/site.config.ts';
</script>

<Seo
	title={SITE.name}
	description="Interactive visualizations of machine learning algorithms: gradient descent and optimizers, with more to come. Runs entirely in your browser."
	path="/"
/>

<section class="hero">
	<p class="eyebrow">Interactive · in-browser · open source</p>
	<h1>See how learning algorithms actually move.</h1>
	<p class="lede">
		Precise, hands-on visualizations of machine learning. Every number is computed live in your
		browser. Change a hyperparameter and watch the dynamics respond.
	</p>
</section>

<section class="grid" aria-label="Visualizers">
	{#each VISUALIZERS as v (v.slug)}
		{@const live = v.status === 'live'}
		<article class="card" class:live>
			<div class="art"><Glyph kind={v.glyph} /></div>
			<div class="body">
				<div class="meta">
					{#if live}
						<span class="badge live-badge">Live</span>
					{:else}
						<span class="badge">Coming soon</span>
					{/if}
					{#each v.tags as t (t)}<span class="tag">{t}</span>{/each}
				</div>
				<h2>
					{#if live}
						<a href={resolve(`/${v.slug}` as '/gradient-descent')} class="stretched">{v.title}</a>
					{:else}
						{v.title}
					{/if}
				</h2>
				<p>{v.description}</p>
			</div>
		</article>
	{/each}
</section>

<style>
	.hero {
		max-width: var(--maxw);
		margin: 0 auto;
		padding: clamp(40px, 8vw, 88px) var(--gutter) clamp(28px, 5vw, 48px);
	}
	h1 {
		font-size: clamp(30px, 5vw, 52px);
		font-weight: 680;
		max-width: 16ch;
		margin-top: 10px;
	}
	.lede {
		max-width: 58ch;
		color: var(--text-2);
		font-size: 17px;
		margin: 16px 0 0;
	}
	.grid {
		max-width: var(--maxw);
		margin: 0 auto;
		padding: 0 var(--gutter);
		display: grid;
		gap: 16px;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 290px), 1fr));
	}
	.card {
		position: relative;
		display: flex;
		flex-direction: column;
		background: var(--surface);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		overflow: hidden;
		transition:
			border-color 0.15s,
			box-shadow 0.15s,
			transform 0.15s;
	}
	.card:not(.live) {
		background: transparent;
	}
	.card:not(.live) .art {
		opacity: 0.55;
		filter: grayscale(0.6);
	}
	.card.live:hover,
	.card.live:focus-within {
		border-color: var(--border-strong);
		box-shadow: var(--shadow);
		transform: translateY(-2px);
	}
	.art {
		padding: 22px 26px 12px;
		border-bottom: 1px solid var(--border);
		background:
			radial-gradient(circle at 1px 1px, var(--border) 1px, transparent 0) 0 0 / 12px 12px,
			var(--surface-2);
	}
	.body {
		padding: 16px 18px 20px;
	}
	.meta {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		margin-bottom: 10px;
	}
	.badge,
	.tag {
		font-family: var(--font-mono);
		font-size: 11px;
		padding: 2px 7px;
		border-radius: 999px;
		border: 1px solid var(--border);
		color: var(--muted);
	}
	.live-badge {
		color: var(--accent);
		border-color: color-mix(in srgb, var(--accent) 45%, transparent);
	}
	h2 {
		font-size: 18px;
		font-weight: 640;
	}
	h2 a {
		color: var(--text);
		text-decoration: none;
	}
	.stretched::after {
		content: '';
		position: absolute;
		inset: 0;
		border-radius: var(--radius);
	}
	.stretched:focus-visible {
		outline: none;
	}
	.stretched:focus-visible::after {
		outline: 2px solid var(--focus);
		outline-offset: 2px;
	}
	p {
		margin: 8px 0 0;
		color: var(--text-2);
		font-size: 14px;
	}
</style>
