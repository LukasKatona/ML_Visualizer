/**
 * Site-wide configuration. Rebranding, the production URL and feature flags live here.
 */

declare const __SITE_URL__: string;

export const SITE = {
	name: 'ML Visualizer',
	tagline: 'Interactive, in-browser visualizations of machine learning algorithms.',
	/** Absolute origin + base path, no trailing slash. Set via SITE_URL at build time. */
	url: typeof __SITE_URL__ === 'string' ? __SITE_URL__ : 'http://localhost:5173',
	/** Optional link to the source repository (shown in the footer when set). */
	repoUrl: '',
	locale: 'en_US'
} as const;

export const FEATURES = {
	/** Render reserved ad slots. Off by default; slots render nothing when false. */
	ads: false
} as const;

export type VisualizerStatus = 'live' | 'soon';
/** Reserved for a future paid tier (with free access via academic institution login). */
export type VisualizerAccess = 'free' | 'premium';

export interface VisualizerMeta {
	slug: string;
	title: string;
	description: string;
	status: VisualizerStatus;
	access: VisualizerAccess;
	tags: string[];
	glyph: 'descent' | 'boundary' | 'attention' | 'xai';
}

export const VISUALIZERS: VisualizerMeta[] = [
	{
		slug: 'gradient-descent',
		title: 'Gradient Descent & Optimizers',
		description:
			'Race SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam and AdamW across classic 2D loss surfaces.',
		status: 'live',
		access: 'free',
		tags: ['optimization', 'Adam', 'momentum'],
		glyph: 'descent'
	},
	{
		slug: 'decision-boundaries',
		title: 'Decision Boundaries',
		description:
			'Train logistic regression and neural networks live; compare k-NN and decision trees on moons, XOR, circles and spirals.',
		status: 'live',
		access: 'free',
		tags: ['classification', 'MLP', 'k-NN'],
		glyph: 'boundary'
	},
	{
		slug: 'attention',
		title: 'Attention',
		description:
			'Scaled dot-product and multi-head attention, step by step: queries, keys, softmax and mixing.',
		status: 'soon',
		access: 'free',
		tags: ['transformers'],
		glyph: 'attention'
	},
	{
		slug: 'xai',
		title: 'Explainability Methods',
		description:
			'Saliency, Integrated Gradients and SHAP side by side on the same model and input.',
		status: 'soon',
		access: 'free',
		tags: ['XAI', 'attribution'],
		glyph: 'xai'
	}
];

export function visualizer(slug: string): VisualizerMeta {
	const v = VISUALIZERS.find((x) => x.slug === slug);
	if (!v) throw new Error(`Unknown visualizer: ${slug}`);
	return v;
}
