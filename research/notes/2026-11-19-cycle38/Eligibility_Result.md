# Cycle 38: the archived trained checkpoint fails the equilibrium gate

19 November 2026. **Decision: abstain from applying cycle 37 at this checkpoint.** Independent calculations pass, but the model is not an equilibrium of either examined update field on the declared input distribution. This is not an instability finding and not a performance experiment. The proposal and backbone remain unchanged.

## Scope fixed before evaluation

The original seed-0 checkpoint from cycle 18 was selected without screening alternatives. The distribution is uniform over its 455 source-training input rows, using only the 13 feature columns and archived input normalization. No targets were numerically parsed, including training targets; no held-out inputs were parsed. The checkpoint itself was previously trained with source labels, which remain part of its inherited information budget. No new labels, optimization, adaptation steps, or accuracy scores were used.

The actual source-shaped model has 1,291 parameters, ReLU hidden layers of widths 30,20,10, and neutral auxiliary value **50**. The audit preserves 50 rather than silently substituting the proposal's abstract 0. Fixed auxiliary constants leave the chain-rule forms unchanged, but the resulting forward predictions depend on the constant. Both student and reference start at the archived weights; reference parameters are frozen during each student gradient, with the input derivative retained in the connected arm.

The official tabular objective is absolute discrepancy. The cycle-37 theory instead addresses the specified smooth squared-loss field. Both are measured here, with connected and stopped backward rules kept separate. This does not validate a squared-loss theorem for the notebook's Adam/absolute-loss algorithm or for an EMA modification of it.

## Results

At matched parameters, the consistency residual has RMS **0.12143227**, maximum absolute value **0.57400215**, and registered near-zero tolerance **2.99649e-8**. The fields are far from the declared stationarity tolerance:

| Field | Euclidean gradient norm | Registered tolerance | Gate |
|---|---:|---:|---|
| Squared loss, connected | 0.24366268 | 1.00000e-8 | fail |
| Squared loss, stopped | 0.53867185 | 2.69110e-8 | fail |
| Absolute loss, connected | 1.56369844 | 8.96767e-8 | fail |
| Absolute loss, stopped | 3.57146603 | 2.97811e-7 | fail |

The numerical scale is parameterization-dependent. Its purpose is to reject proximity to zero under the prespecified tolerance, not rank algorithms by usefulness. A nonzero residual alone need not exclude a stationary point; the directly computed nonzero fields are the decisive evidence here. Since the reference velocity is zero at matched weights but the student velocity is not, the joint state is not an EMA equilibrium either.

Minimum absolute hidden preactivations are 2.01725e-4 on the student pass and 6.17152e-6 on the reference pass. Neither has values within the registered 1e-10 switching-surface diagnostic on these inputs. Thus the failed stationarity gate cannot be attributed simply to an exactly zero hidden preactivation in this audit. No global smoothness or activation-region radius is inferred.

As registered, **H,K,L and their spectra were not computed** after the equilibrium gate failed. In particular, symmetrizing a nonstationary checkpoint's curvature would not turn it into a valid equilibrium certificate. This finding concerns one checkpoint on one empirical source-input distribution; it does not rule out other equilibria, local stability elsewhere, or behavior on unseen test distributions.

## Parameter rank and a genuine redundancy

The 455-by-1,291 student-output Jacobian has **numerical rank 167** at singular-value tolerance 6.10819e-8. Its numerical nullity is therefore 1,124; this is a finite-input, tolerance-dependent diagnostic, not an exact global rank theorem. The JSON key `parameter_nullity_lower_bound` refers to this numerical calculation, not a rigorous algebraic lower bound. At an exact zero-residual state, the reduced full-input matrix J^T diag((1-A)^2) J cannot have rank greater than J. That reduced formula is not the full Hessian at this nonzero-residual checkpoint.

A separate structural check identifies one real parameter redundancy. For the first hidden unit, multiply its incoming weights and bias by c>0 and divide the next layer's corresponding outgoing column by c. Because ReLU(c*a)=c*ReLU(a), the entire network function remains unchanged for every input and auxiliary value. With c=1.1, the measured maximum prediction discrepancy across both passes is 4.44e-16. The infinitesimal rescaling direction has norm 1.08799607, while J times that direction is zero to reported precision.

This verifies one known functional redundancy. It does **not** establish that all 1,124 numerical null directions can be removed without altering dynamics or future predictions. A valid observable-state reduction needs additional invariance/closure analysis. Nor does a function-preserving parameter transformation make Euclidean gradient dynamics equivariant away from equilibria; no neutral eigenvalue claim at this nonstationary point is made.

## Reproduction and integrity

`PROTOCOL.md`, the script, and all input hashes were recorded in `registered.json` before the audit executed. An initial registration command pointed to the cycle-15 notebook path, which was only a manifest location. That command failed before any evaluation; the actual archived cycle-9 notebook was then located and registered. The protocol and numerical thresholds were unchanged.

`audit_eligibility.py` computes per-example derivatives independently with NumPy layerwise reverse differentiation, then compares the resulting four parameter fields against PyTorch automatic differentiation with matched weights and explicit graph choices. Maximum field discrepancy: **1.44329e-15**. Maximum independent forward discrepancy: **5.55112e-16**. All conformance checks passed. Runtime was **0.109375 CPU seconds excluding imports**, below the 30-second cap, using NumPy 2.1.3 and PyTorch 2.9.0+cpu on one thread. Results, singular values, predictions, residuals and gradients are retained in `results.json` and `diagnostics.npz`; no label arrays are included.

Reproduce with:

    python audit_eligibility.py --repo C:\Users\amita\source\repos\Research_Repo

Quote the path if needed in another shell. A rerun rewrites the local diagnostic outputs; preserve the recorded run if comparing timings.

## Primary evidence and interpretation

- Official implementation, revision **100524001d890efc08911f3cba8f45f51f4aa7d1**, notebook `exps/tabular/vanilla_model.ipynb`: https://github.com/nikitadurasov/ittt/blob/100524001d890efc08911f3cba8f45f51f4aa7d1/exps/tabular/vanilla_model.ipynb . Local archived notebook SHA-256 **6d583591e7433547c3b8fbb85f2177e51a933596f21709bc30b251be89c1e227**, matching the prior source manifest. Reinspected zero-based cell 0, source lines 34-44 (auxiliary default and ReLU forward); cell 12 line 9 (k=50); cell 13 lines 36-38 (prediction feedback and absolute discrepancy), line 42 (student reset), lines 52 and 64 (backward and Adam). The new diagnostic is local code, not execution of the notebook.
- Durasov et al., *IT3: Idempotent Test-Time Training*, ICML/PMLR 267, pp.14867-14883, 2025: https://proceedings.mlr.press/v267/durasov25a.html , accessed 19 November 2026. This is the paper/implementation provenance, not a source for today's numerical findings or an assertion that its experiments satisfy the new theorem.
- Local theoretical assumptions: current proposal Sections 3 and 4.2; backbone `The specified dynamics`; cycle-37 `Restoring_Geometry_Result.md`, especially its equilibrium and H,L hypotheses. No new external theorem is claimed by this audit.

## Decision and next work

Retain cycle 37 as an analytic positive control. Do not present its learning-rate threshold as a certified bound for this trained checkpoint. The submitted proposal already distinguishes stationary certificates from nonstationary diagnostics, so no submission revision is warranted.

Before enlarging the theory branch or doing another model sweep, close the pending closest-prior-art comparison with residual versus semi-gradient methods (cycle 35 inspected only Baird's abstract). Compare the exact backward rule and idempotent shared-architecture separation, including what is merely a classical gradient-field consequence. The empirical graph/optimizer attribution remains the leading thesis evidence. Publication novelty and trained-model usefulness are still unresolved.
