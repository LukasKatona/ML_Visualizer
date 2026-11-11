<script lang="ts">
	import { SITE } from '#lib/site.config.ts';

	interface Props {
		title: string;
		description: string;
		/** Route path relative to the site root, e.g. '/' or '/gradient-descent/'. */
		path: string;
		image?: string;
	}

	let { title, description, path, image = '/og/default.png' }: Props = $props();

	const fullTitle = $derived(title === SITE.name ? title : `${title} · ${SITE.name}`);
	const canonical = $derived(SITE.url + path);
	const imageUrl = $derived(SITE.url + image);
</script>

<svelte:head>
	<title>{fullTitle}</title>
	<meta name="description" content={description} />
	<link rel="canonical" href={canonical} />
	<meta property="og:type" content="website" />
	<meta property="og:site_name" content={SITE.name} />
	<meta property="og:locale" content={SITE.locale} />
	<meta property="og:title" content={fullTitle} />
	<meta property="og:description" content={description} />
	<meta property="og:url" content={canonical} />
	<meta property="og:image" content={imageUrl} />
	<meta property="og:image:width" content="1200" />
	<meta property="og:image:height" content="630" />
	<meta name="twitter:card" content="summary_large_image" />
	<meta name="twitter:title" content={fullTitle} />
	<meta name="twitter:description" content={description} />
	<meta name="twitter:image" content={imageUrl} />
</svelte:head>
