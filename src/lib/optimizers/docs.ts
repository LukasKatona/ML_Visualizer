/**
 * Update rules (KaTeX source) and concise notes per optimizer. The KaTeX is rendered to HTML at
 * build time (see routes/gradient-descent/+page.server.ts), so no KaTeX JS ships to the client.
 *
 * Notation: g_t = ∇f(θ_t) (possibly noisy), operations on vectors are element-wise.
 */
import type { OptimizerId } from './optimizers.ts';

export interface OptimizerDoc {
	rule: string;
	good: string;
	bad: string;
}

export const OPTIMIZER_DOCS: Record<OptimizerId, OptimizerDoc> = {
	sgd: {
		rule: String.raw`\theta_{t+1} = \theta_t - \eta\, g_t`,
		good: 'Well-conditioned problems. With a decaying η it converges under gradient noise (Robbins–Monro), and in deep nets its noise acts as an implicit regulariser.',
		bad: 'Ill-conditioning: stable only for η < 2/L, and the best rate on a quadratic is (κ−1)/(κ+1). It crawls on plateaus and zig-zags across narrow valleys.'
	},
	momentum: {
		rule: String.raw`\begin{aligned} v_{t+1} &= \beta\, v_t + g_t \\ \theta_{t+1} &= \theta_t - \eta\, v_{t+1} \end{aligned}`,
		good: 'Builds up velocity along consistent low-curvature directions and damps oscillation across ravines. Tuned on a quadratic, the rate improves to (√κ−1)/(√κ+1).',
		bad: 'Large β overshoots and rings around the minimum. Stable for ηL < 2(1+β). Acceleration is not guaranteed on general convex functions.'
	},
	nesterov: {
		rule: String.raw`\begin{aligned} v_{t+1} &= \beta\, v_t + \nabla f(\theta_t - \eta\beta\, v_t) \\ \theta_{t+1} &= \theta_t - \eta\, v_{t+1} \end{aligned}`,
		good: 'The look-ahead gradient corrects the velocity before it overshoots, which gives less ringing than heavy-ball momentum and the optimal O(1/t²) rate on smooth convex problems.',
		bad: 'Tighter stability limit on quadratics: ηL < 2(1+β)/(1+2β), so it diverges at learning rates where heavy-ball is still stable. Under gradient noise most of the acceleration disappears.'
	},
	adagrad: {
		rule: String.raw`\begin{aligned} G_t &= G_{t-1} + g_t^{2} \\ \theta_{t+1} &= \theta_t - \frac{\eta\, g_t}{\sqrt{G_t} + \epsilon} \end{aligned}`,
		good: 'Sparse features and convex online learning. Per-coordinate scaling handles axis-aligned ill-conditioning without tuning.',
		bad: 'G only grows, so the effective step decays like 1/√t and long non-convex runs stall (compare Rosenbrock).'
	},
	rmsprop: {
		rule: String.raw`\begin{aligned} v_t &= \beta\, v_{t-1} + (1-\beta)\, g_t^{2} \\ \theta_{t+1} &= \theta_t - \frac{\eta\, g_t}{\sqrt{v_t} + \epsilon} \end{aligned}`,
		good: "An EMA of g² replaces AdaGrad's sum, so the step size doesn't decay to zero. Works well for non-stationary objectives.",
		bad: 'Near the optimum, steps ≈ η·sign(g), so without decaying η it settles into a limit cycle of size ~η. It has no momentum, and the preconditioning is diagonal only.'
	},
	adam: {
		rule: String.raw`\begin{aligned} m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t \\ v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, g_t^{2} \\ \hat m_t &= \tfrac{m_t}{1-\beta_1^{t}},\quad \hat v_t = \tfrac{v_t}{1-\beta_2^{t}} \\ \theta_{t+1} &= \theta_t - \frac{\eta\, \hat m_t}{\sqrt{\hat v_t} + \epsilon} \end{aligned}`,
		good: 'A robust default. Invariant to gradient scale, so it copes with both plateaus and steep walls (Beale). Bias correction makes the first step exactly η·sign(g).',
		bad: 'Can fail to converge on simple convex problems (Reddi et al., 2018). Diagonal preconditioning does not fix rotated ill-conditioning. An L2 penalty added to the loss gets rescaled by 1/√v̂ (hence AdamW).'
	},
	adamw: {
		rule: String.raw`\begin{aligned} m_t, v_t, \hat m_t, \hat v_t &\ \text{as in Adam} \\ \theta_{t+1} &= \theta_t - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda\, \theta_t\right) \end{aligned}`,
		good: 'Decoupled weight decay: shrinkage is independent of the adaptive scaling (Loshchilov & Hutter, 2019). The standard choice for transformers.',
		bad: 'Here the decay pulls toward the origin, so when the minimum is not at 0 the fixed point shifts away from it (Rosenbrock, Beale). The shift is visible at large λ.'
	}
};
