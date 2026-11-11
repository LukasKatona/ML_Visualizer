/** Model formulas (KaTeX, rendered at build time) and concise notes for the decision-boundary viz. */
import type { ModelId } from './session.ts';

export interface ModelDoc {
	tex: string;
	good: string;
	bad: string;
}

export const MODEL_DOCS: Record<ModelId, ModelDoc> = {
	logreg: {
		tex: String.raw`\begin{aligned} p(y{=}1\mid x) &= \sigma\!\left(w^\top \varphi_d(x) + b\right) \\ \mathcal L &= -\tfrac1n\textstyle\sum_i \big[y_i\log p_i + (1-y_i)\log(1-p_i)\big] \\ &\quad + \tfrac{\lambda}{2}\lVert w\rVert^2 \end{aligned}`,
		good: 'Convex: any optimizer that converges reaches the global optimum, and the boundary is a level set of a degree-d polynomial. Well-calibrated probabilities near the data.',
		bad: 'On separable data with λ = 0, ‖w‖ → ∞: the boundary converges in direction to the max-margin solution while p saturates. High degree without λ gives wild boundaries away from the data.'
	},
	mlp: {
		tex: String.raw`\begin{aligned} h_0 &= \varphi_d(x),\quad h_\ell = a\!\left(W_\ell h_{\ell-1} + b_\ell\right) \\ p(y{=}1\mid x) &= \sigma\!\left(w_L^\top h_{L-1} + b_L\right) \\ a &\in \{\tanh, \mathrm{ReLU}\} \end{aligned}`,
		good: 'Learns its own features: one hidden layer already solves XOR and moons. Depth lets it compose features, which helps on spirals.',
		bad: 'Non-convex: the result depends on the initialisation (the seed) and the learning rate. ReLU nets give piecewise-linear boundaries and can lose units that never activate (dead ReLUs); tanh nets train more slowly at large width.'
	},
	knn: {
		tex: String.raw`\hat p(y{=}1\mid x) = \frac1k \sum_{i \in \mathcal N_k(x)} \mathbf 1[y_i = 1]`,
		good: 'No training and no assumptions about the boundary. As n → ∞ with k → ∞ and k/n → 0 it is consistent (it approaches the Bayes classifier).',
		bad: 'k = 1 memorises noise (train accuracy 1, jagged boundary); large k over-smooths. Every prediction scans the whole training set, and distances lose meaning in high dimensions.'
	},
	tree: {
		tex: String.raw`\begin{aligned} G(S) &= 2\,p_S(1-p_S) \\ (j^\ast, t^\ast) &= \arg\max_{j,t}\; G(S) - \tfrac{|S_L|}{|S|}G(S_L) - \tfrac{|S_R|}{|S|}G(S_R) \end{aligned}`,
		good: 'Axis-aligned, piecewise-constant, interpretable boundaries. Insensitive to feature scaling; deep trees reach zero training error.',
		bad: 'Diagonal or curved boundaries become staircases. Deep trees have high variance (small changes in the data give different splits), which is why ensembles exist.'
	}
};
