import katex from 'katex';
import { SURFACES, SURFACE_IDS, type SurfaceId } from '#lib/math/surfaces.ts';
import { OPTIMIZER_DOCS } from '#lib/optimizers/docs.ts';
import { OPTIMIZER_IDS, type OptimizerId } from '#lib/optimizers/optimizers.ts';

// Rendered once at build time (the page is prerendered), so the client never loads KaTeX JS.
const render = (tex: string) =>
	katex.renderToString(tex, { displayMode: true, throwOnError: true });

export function load() {
	return {
		tex: {
			optimizers: Object.fromEntries(
				OPTIMIZER_IDS.map((id) => [id, render(OPTIMIZER_DOCS[id].rule)])
			) as Record<OptimizerId, string>,
			surfaces: Object.fromEntries(
				SURFACE_IDS.map((id) => [id, render(SURFACES[id].latex)])
			) as Record<SurfaceId, string>
		}
	};
}
