import katex from 'katex';
import { MODEL_DOCS } from '#lib/ml/docs.ts';
import { MODEL_IDS, type ModelId } from '#lib/ml/session.ts';

// Rendered once at build time (the page is prerendered), so the client never loads KaTeX JS.
export function load() {
	return {
		tex: Object.fromEntries(
			MODEL_IDS.map((id) => [
				id,
				katex.renderToString(MODEL_DOCS[id].tex, { displayMode: true, throwOnError: true })
			])
		) as Record<ModelId, string>
	};
}
