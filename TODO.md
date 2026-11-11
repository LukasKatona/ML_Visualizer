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

## Product

- [ ] Set `SITE.repoUrl` in `src/lib/site.config.ts` (footer source link).
- [ ] Custom domain (see README → Deployment).
- [ ] Paid tier + academic institution login (see README → Roadmap; needs a backend or auth
      provider).
