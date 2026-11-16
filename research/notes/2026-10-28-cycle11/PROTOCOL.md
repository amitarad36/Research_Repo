# Next bounded analysis: discrete sign updates and their limiting interpretation

Saved 27 October 2026 after the finite absolute-loss experiment was classified numerically unresolved. This is a theoretical diagnostic prompted by that failure, not a new confirmatory experiment or permission to retune it.

## One question

Does a pointwise sign(0)=0 ODE provide a well-posed continuous-time model at an attracting residual-zero surface in the matched-reference system, and what can be proved instead about its explicit discrete updates? Settle this before using smooth Lyapunov or linearization statements across sign changes.

## Fixed model and task

Use one scalar dual-input affine model f_theta(z)=h+theta+a*z, with h=1,a=1/4, theta_0=phi_0=0. The evaluator target is h, so initialization is correct and nonzero; correctness and a are planted. q_theta=f_theta(0)=h+theta, cached input q_0=h. This diagnostic is not the learned model, an idempotence-trained solution or a neural explanation.

Derive original stopped-output, original full input-gradient and cached-input fields for D=abs(f_phi(z)-q_theta), with z=q_theta or q_0 as declared and sign(0)=0. Reproduce the local stopped/cached sign-region identity. The cached residual is phi-theta+c with c=a*h=1/4. Fixed reference and student-first EMA with tau=1 are the two schedules; alpha=tau/(tau+eta). All calculations must derive from this one forward graph.

## Required analysis and checks

1. Derive exact fixed-reference and EMA discrete recurrences. Use e=theta-phi to test the candidate cached recurrence e_next=alpha*(e+eta*sign(c-e)); verify it algebraically, not by fitting trajectories. Keep actual step-size dependence explicit.
2. For fixed weights, classify finite-step arrival or cycling under constant eta, including exact-zero initialization and hitting cases. Check a few rational iterates for eta=1/10; no neural updates or eta sweep. This is an illustration, not a replacement result for cycle 10.
3. Analyze the pointwise ODE and an explicitly defined set-valued sign extension at the residual-zero surface. Prove or refute existence/uniqueness with precise solution notion and assumptions; do not silently replace sign(0)=0 by a selected sliding velocity. Verify applicable primary mathematical sources and distinguish their established theorem from a new application calculation.
4. Attempt a finite-horizon bound connecting the discrete cached EMA recurrence to the justified limiting evolution, with constants, eta restrictions and behavior at the first boundary. If the bound fails, preserve a counterexample or the precise gap. Do not assert convergence from two step sizes or infer a neural rate from this affine example.
5. State what survives for the other backward rules, only as implied by the same algebra. Do not generalize a zero local teacher derivative to all rules, or a scalar relay calculation to multivariate nonlinear feedback.

## Budget, rejection and scope

One analytic cycle; exact algebra and <=30 CPU seconds of scalar/rational verification if helpful. No new trained checkpoints, adaptation trajectories on the saved neural data, loss smoothing, alternative horizons, parameter sweeps, paid compute or packages. Do not extract labels. The 60 neural trajectories and their numerical-failure classification remain immutable.

Success means a checked, assumption-explicit statement identifying a valid continuous/discrete interpretation or a rigorous obstruction. If only standard subgradient facts are obtained, record them as background and reassess the paper direction rather than inflating novelty. No risk-benefit, current-task correctness after adaptation, retention or recovery conclusion follows without its own assumptions. The goal is to settle a foundational gap, not to explain the observed neural sensitivity merely by analogy.
