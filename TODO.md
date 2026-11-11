# TODO

## ML review (deferred)

Choices made during the build that need an ML sanity check.

### Gradient Descent & Optimizers

- [ ] **Quadratic conditioning is axis-aligned** (`f = ½(x² + 25y²)`). That's the easy case for
      AdaGrad/RMSProp/Adam, which scale each coordinate separately. Should there be a rotated
      variant where that scaling doesn't help? (`src/lib/math/surfaces.ts`)
- [ ] **Gradient noise model** is additive isotropic Gaussian, `g̃ = ∇f + σξ`. Alternative: true
      minibatch noise from a 2-parameter regression on sampled data, where the noise depends on
      the current parameters. (`src/lib/sim/engine.ts`)
- [ ] **RMSProp default β = 0.9** (Hinton's lecture value) vs. PyTorch's 0.99.
      (`src/lib/optimizers/optimizers.ts`)
- [ ] **AdamW weight decay pulls toward the origin**, so on Rosenbrock/Beale the fixed point is
      biased away from the true minimum. Keep it as a teaching point, or set default λ = 0 on
      surfaces whose minimum is not at 0?
- [ ] Review the default learning rates and start points per surface, and the optimizer notes
      ("Works well" / "Struggles") in `src/lib/optimizers/docs.ts`.

### Decision Boundaries

- [ ] **Dataset definitions and default noise** (σ = 0.15, 70/30 split): are moons/spirals scaled
      sensibly, and is the XOR margin (points kept ≥ 0.08 from the axes) right?
      (`src/lib/ml/datasets.ts`)
- [ ] **Polynomial features are not standardised.** At degree 5, x⁵ spans about ±10 while x spans
      ±1.6, which skews plain SGD. Standardise the features, or leave the effect visible as a
      lesson? (`src/lib/ml/mlp.ts`)
- [ ] **Initialisation:** Xavier-normal for tanh, He-normal for ReLU, output layer Xavier, zero
      biases.
- [ ] **L2 is applied to weights only** (not biases), as λ/2·‖W‖² added to the mean BCE.
- [ ] **k-NN ties:** with even k, P = 0.5 exactly lies on the boundary; no tie-breaking rule.
- [ ] Model notes ("Works well" / "Struggles") in `src/lib/ml/docs.ts`.

## Product

- [ ] Set `SITE.repoUrl` in `src/lib/site.config.ts` (footer source link).
- [ ] Custom domain (see README → Deployment).
- [ ] Paid tier + academic institution login (see README → Roadmap; needs a backend or auth
      provider).
