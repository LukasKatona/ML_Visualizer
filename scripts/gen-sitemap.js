// Post-build: write sitemap.xml and robots.txt into build/ using absolute URLs from SITE_URL.
// Pages are discovered from the prerendered output, so new visualizers are picked up automatically.
import { readdirSync, statSync, writeFileSync } from 'node:fs';
import { join, relative, sep } from 'node:path';

const OUT = 'build';
const base = process.env.BASE_PATH ?? '';
const site = (process.env.SITE_URL ?? `http://localhost:5173${base}`).replace(/\/$/, '');

/** @param {string} dir @returns {string[]} */
function htmlFiles(dir) {
	return readdirSync(dir).flatMap((name) => {
		const p = join(dir, name);
		if (statSync(p).isDirectory()) return name === '_app' ? [] : htmlFiles(p);
		return name === 'index.html' ? [p] : [];
	});
}

const today = new Date().toISOString().slice(0, 10);
const urls = htmlFiles(OUT)
	.map((f) => {
		const dir = relative(OUT, f).split(sep).slice(0, -1).join('/');
		return dir ? `/${dir}/` : '/';
	})
	.sort();

const sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls.map((u) => `\t<url><loc>${site}${u}</loc><lastmod>${today}</lastmod></url>`).join('\n')}
</urlset>
`;

writeFileSync(join(OUT, 'sitemap.xml'), sitemap);
writeFileSync(join(OUT, 'robots.txt'), `User-agent: *\nAllow: /\n\nSitemap: ${site}/sitemap.xml\n`);
console.log(`sitemap: ${urls.length} URLs under ${site}`);
