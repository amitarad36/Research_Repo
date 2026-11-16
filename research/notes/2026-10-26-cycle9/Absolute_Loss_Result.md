# Absolute loss changes the local feedback explanation

26 October 2026, cycle 9. **New checked derivation and local numerical audit**, plus a recheck of one exact official code path. The stopped-output/cached-input local identity is verified in all five checkpoints. This narrows the earlier squared-loss explanation; it does not show absence of feedback in the original full input-gradient rule or equality of complete nonlinear trajectories.

## Exact source evidence and transfer limits

Re-read the saved IT3 audit and retrieved `exps/tabular/vanilla_model.ipynb` at revision **100524001d890efc08911f3cba8f45f51f4aa7d1**. Notebook SHA256: `6d583591e7433547c3b8fbb85f2177e51a933596f21709bc30b251be89c1e227`. The complete notebook, numbered code-cell extraction and URL/hash manifest are preserved; no notebook code was executed. Source retrieval followed web-fetch cache misses and a sandbox DNS failure; an approved read-only network request succeeded.

[Pinned official notebook](https://github.com/nikitadurasov/ittt/blob/100524001d890efc08911f3cba8f45f51f4aa7d1/exps/tabular/vanilla_model.ipynb), zero-based JSON cells and one-based source lines within each cell:

- Cell 13, lines 34-38: the student prediction is supplied to the reference without detach; discrepancy is `.abs().mean()`, meaning a mean over all output entries, not the sum over output coordinates followed only by a batch mean.
- Cell 13, lines 41-55: student weights reset from the reference per batch, student train mode/reference evaluation mode, loss backward, and optimizer step. Evaluation mode does not itself disable input gradients.
- Cell 13, lines 61-64 and 87-103: Adam contains only the student parameters, is constructed outside the batch loop, and is called for two steps at learning rate 1e-4. Weight reset does not reset its state. No EMA update appears in this selected cell. This is not evidence for a reproducible official online EMA schedule.
- Cell 5, lines 82-109: the later ZigZag class concatenates x and y directly, then uses three ReLU hidden layers; no detach occurs on the supplied y in this forward path. Cell 12, lines 5-9 instantiates one output and `k=50`. When y is omitted, cell 5, lines 98-104 uses that constant auxiliary input, not zero. The audit reads the notebook as stored; it does not certify a historical interactive execution order.

Our five checkpoints remain the explicitly chosen two-output tanh model with zero auxiliary input, squared source pretraining, GD and a student-first EMA. Only the discrepancy reduction/backward-path distinction is being transferred here. Architecture, default auxiliary input, source training and optimizer/reset policies differ. The official notebook is not reproduced, and a universal IT3 graph is not asserted.

We define sign(0)=0 in the mathematical/numerical fields. This is consistent with the minimum-norm subgradient rule for convex nonsmooth elementary operations in [PyTorch 2.14 autograd documentation, “Gradients for non-differentiable functions”](https://docs.pytorch.org/docs/2.14/notes/autograd.html#gradients-for-non-differentiable-functions). The notebook's historical PyTorch version is not identified and was not executed. A subgradient convention does not supply a classical derivative at a zero residual. All numerical derivatives below stay away from zero; ReLU activation boundaries would impose additional restrictions in the official architecture.

## Precise fields

For a fixed finite batch of N samples and d outputs, set `q=q_theta(x)`, `r=T_phi(x,q)-q`, `J=d_theta q`, `A=d_q T_phi`, `B=I-A`, and

`D1 = (1/(N*d)) sum_{i,o} |r_io|`.

Writing sigma=sign(r), the **negative** student-gradient fields, with reference parameters excluded from the student update, are

`F_input = (1/d) mean_i J_i^T B_i^T sigma_i`,

`F_stop = (1/d) mean_i J_i^T sigma_i`.

For a cache `c_i=q_theta0(x_i)` held fixed, let `r_c=T_phi(x,c)-q` and sigma_c=sign(r_c). Then

`F_cached = (1/d) mean_i J_i^T sigma_c,i`.

Both backward choices coincide for the cached-input control, whose target has no student dependence. The factor 1/d is required by the notebook's mean over all entries: it is 1/2 for our d=2 model, while the selected notebook has d=1. No normalization, smoothing, learning-rate adjustment or velocity matching was introduced. The stopped-output rule remains a controlled comparison, not the audited original regression backward path.

## Local identity and proof

Assume q and T are C2 on the parameter/input neighborhood being considered. At matched initialization with c=q_theta0, suppose every residual component is nonzero. Since the batch is finite and the maps are continuous, there is an open joint-parameter neighborhood in which the original and cached residuals share the same fixed sign array sigma0. On this neighborhood,

`F_stop(theta,phi) = F_cached(theta,phi) = (1/d) mean J_theta^T sigma0`.

Thus the fields agree throughout that neighborhood, not merely at the initial point. Their full derivative blocks are

`S_stop = S_cached = (1/(N*d)) sum_{i,o} sigma0_io Hessian_theta(q_io)`,

`T_stop = T_cached = 0`, where S=d_theta F and T=d_phi F.

The residual-weighted squared-loss term `mean J^T A J` is absent because the derivative of the sign is zero within a fixed nonzero-sign region. Prefactor derivatives do not vanish: the weighted output Hessian remains. This is a local calculus result, not a new general theorem about continual adaptation. It does not require A=0.

Matched stopped/cached discrete trajectories with the same deterministic optimizer/reference update remain equal for as long as their residual sign arrays agree at their shared states. A later sign disagreement can change the velocities; cancellation can also make different signs give the same weighted update. Equality of initial derivative blocks does not preclude a finite-time intervention effect. The current cycle has not run any absolute-loss trajectory or observed such a crossing.

## Full input-gradient feedback remains

Inside a fixed-sign region, write `H_q(w)[u] = (1/d) mean (d_theta J[u])^T w` for fixed output weights w, and let `A_theta[u]` differentiate A through q_theta while phi is fixed. For a reference perturbation v, `A_phi[v]` holds q fixed. The full blocks are

`S_input[u] = H_q(B^T sigma0)[u] - (1/d) mean J^T A_theta[u]^T sigma0`,

`T_input[v] = -(1/d) mean J^T A_phi[v]^T sigma0`.

Both the student-Jacobian prefactor and the reference-input Jacobian must be differentiated. A locally constant residual sign does **not** remove these terms. In the audited smooth checkpoints the input-gradient teacher block is nonzero in every seed. Therefore “absolute loss has no local teacher feedback” would be false; the zero teacher block above belongs to the stopped/cached comparison under its sign assumptions.

## Fixed-protocol numerical audit

Reused all five source/data/hash-matched checkpoints and the same 64 unlabeled inputs, d=2; no labels extracted, source training or adaptation updates. Primary seeds 0,2,3,4, with 1 sensitivity-only. Protocol SHA256 `8bcf5082b1f334bbb94e7cf9ed9a23ce52427844a2553ef6887d118195f94ff6` is byte-identical to cycle 8's next-audit protocol.

| Seed | Minimum initial absolute residual | Maximum gradient/block error | Full input-gradient teacher-block Frobenius norm |
|---|---:|---:|---:|
| 0 | 1.22062e-4 | 6.068e-12 | .66524 |
| 2 | 3.01458e-5 | 1.007e-11 | .44945 |
| 3 | 4.83910e-4 | 9.090e-12 | .58412 |
| 4 | 4.28933e-5 | 6.914e-12 | .75576 |
| 1, sensitivity | 2.54963e-4 | 8.324e-12 | .84695 |

All 640 initial residual entries are nonzero and exceed 1e-10 in magnitude. Central differences used h=1e-5 in all 114 student and all 114 reference coordinates, for all three rules; no step reductions were needed. The preliminary probes and every subsequent loss/field finite-difference evaluation preserved signs. Across 10,260 derivative/loss residual evaluations the minimum margin was 2.01458e-5. These checks cover the evaluated coordinate probes; they are not a quantitative certificate for an entire ball.

Full loss-gradient checks and independently assembled product-rule blocks agree within **1.007e-11**, below the fixed 1e-7 tolerance. Stopped/cached student blocks match exactly in the finite-difference arrays, as do the velocities; both reference blocks are exactly zero. Full input-gradient reference-block norms .44945-.84695 establish the contrasting nonzero terms. A is nonzero in every seed.

`verify_saved.py` independently assembles the sign-weighted output Hessian directly from tanh first/second derivatives and cross-parameter terms; maximum difference from the computed Hessian is 8.327e-17. Explicit per-sample velocity means agree within 4.164e-17. Full input-gradient student-block symmetry error is <=5.552e-17, consistent with its being a negative loss Hessian inside the region. All archive/source/script hashes are checked. Main computation: 7.515625 CPU seconds, 8.82423 wall seconds, below the 30-second target. No installs, GPU, paid compute or cluster work.

Results SHA256: `2e6be33181896316189efae3829abdf352f492402e734c084bdfed8c15ee2043`. Full provenance, finite-difference steps, sign margins and checks are in `absolute_loss_results.json`; five archives retain all full blocks, product-rule pieces, gradients, residuals and signs. The completed script refuses to overwrite results. The source notebook is evidence only; it is never imported or executed by the audit.

## Decision

The earlier successful initial-affine stopped/cached forecast is specific to the squared surrogate's smooth residual weighting; its local input-path block difference does not transfer unchanged to absolute discrepancy. Keep the squared result as checked evidence, but withdraw any inference that those same matrices explain the selected official absolute-loss dynamics. Full input-gradient feedback remains an eligible mechanism, and sign-region transitions are a candidate explanation for a finite stopped/cached difference.

The narrowed thesis question is now **how loss derivatives and the backward rule determine which reference-feedback effects are visible locally and which require crossing loss boundaries**. No novelty is established: this audit is a necessary scope correction, not a first paper by itself. A paper would still need a substantive prediction or guarantee beyond elementary piecewise-smooth calculus and an implementation-relevant intervention.

Next single action: the fixed finite-time absolute-loss intervention in `NEXT_EXPERIMENT_PROTOCOL.md`, with paired sign/state records, same declared memory and update budgets, separate numerical and effect gates, and no smoothing or scale tuning. No results from that experiment exist yet.
