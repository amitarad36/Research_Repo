# Cycle 3: learned-geometry gate

Written before training/results, 22 October 2026. This is a bounded synthetic mechanism test, not a reproduction of an IT3 benchmark or a novelty claim.

## Question and prediction

Does ordinary zero/label supervised pretraining learn the strong rotational reference-input geometry used by the previous constructed example, near useful predictions? No geometry regularizer, injected feedback matrix, label-based checkpoint selection, architecture search, or seed selection is allowed. The prior construction predicts a backward-rule/EMA stability separation only when its geometry and parameterization assumptions hold. Its empirical relevance is unestablished; this test can reject its use as the leading explanation for this model.

## Fixed budget and model

- NumPy float64 CPU, five seeds 0,1,2,3,4. One tanh hidden layer, 4 inputs (x,z), 16 hidden units, 2 linear outputs: 114 parameters. All parameters trained and included in adaptation derivatives. No BatchNorm, dropout, weight decay, replay, source regularizer, or optimizer state in adaptation.
- Shared training data: 256 iid inputs uniform on [-1,1]^2, RNG seed 314159. Independent evaluation data: 256 such inputs, seed 271828. Fixed labels y1=1+0.3*x1+0.15*x1*x2, y2=0.8+0.3*x2+0.15*sin(pi*x1). Labels are nonconstant and nonzero throughout the domain.
- Train on equal-weight pairs (x,0)->y and (x,y)->y, using squared error (the declared smooth surrogate). 6000 minibatch updates, 128 source examples plus both auxiliary versions. Adam beta1=.9, beta2=.999, epsilon=1e-8; learning rate .001 for 4500 updates, .0003 for 1500. Xavier-normal weights, zero biases. Save final checkpoint, no early stopping or retries.
- The recipe is motivated by IT3 arXiv:2410.04201v2, Section 3.1, Eq. (2), with the resulting composition in Eq. (4), https://arxiv.org/html/2410.04201v2 . The tiny architecture, label function, optimizer schedule, squared training loss/discrepancy, and diagnostics are project choices. Official regression code uses absolute discrepancy; this is not its reproduction.

## Gates fixed before results

1. Fit gate: held-out sum-squared prediction error divided by held-out total label variance <= .05 for at least 3/5 seeds. Report zero-input and label-input errors separately; both must pass for an eligible seed. If fewer pass, the geometry test is inconclusive due to insufficient fit; do not tune within this cycle.
2. Rotation gate: at both auxiliary z=q and z=y, at least 10% of evaluation points in at least 3 eligible seeds have eigenvalues a +/- i*w with 0<a<1 and w^2 > a*(1-a) (strict numerical margin 1e-6). This is the strong-rotation regime of the normal 2D family A=a*I+w*J: its fast-reference input-gradient output model contracts while the stopped rule expands. Applying the spectral pattern to a general nonnormal A is only a screening heuristic, not a neural stability criterion. Failure rejects prioritizing that specific strong-rotation explanation here, not all possible backward-rule differences.
3. Report ||A||_2, ||(A-A^T)/2||_F, ||A^2-A||_F, normalized idempotence defect ||A^2-A||_F/(||A||_F+1e-12), complex-eigenvalue fraction, and the rotation fraction. Do not label a small absolute defect as projection geometry if A is merely small.
4. Full parameter check on the first 64 evaluation inputs, labels excluded from the field: central-difference derivatives of the actual student velocity with respect to all student and reference parameters, separately for input-gradient and output-detached rules. Assemble the 228x228 coupled ODE linearization at matched final checkpoints for tau=1,20 and student-first discrete derivative eta=.1, alpha=tau/(tau+eta). Check a second finite-difference step and directional finite differences. Measure nonzero velocity/residual explicitly. A trained checkpoint need not be an equilibrium, so these spectra are local diagnostics, not stability certificates. Include output sensitivity and parameter degeneracy in interpreting near-zero modes.
5. No stream/adaptation sweep this cycle if rotation gate fails. Preserve the null and revise the next question based on measured geometry. Any next experiment requires a new prediction and budget. Labels used at z=y are evaluation-only derivative probes after pretraining, never supplied to the adaptation field.

## Verification

Before interpreting results: directional finite differences for supervised gradient, input Jacobian, and both backward rules (for detached output, hold the reference target fixed during the student loss finite difference). Compare full-field derivatives at eps=1e-5 and 5e-6. Save parameters, shared data, code, protocol, JSON summaries, and environment versions. Record runtime and all five seeds, including failures.
