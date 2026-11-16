# When can the cached reference predict a second gradient change?

5 November 2026, cycle 27. Exact scalar-output calculation for the current-connected and cached-connected graphs after the same first Adam step. No training or held-out inputs or labels used.

## Inputs and proof

Fix the feature vector, frozen weights, and cache c. Compute all preactivations u at c. The first slope vector v is the prediction-input column of the first weight matrix. At each later layer propagate v as W diag(1[u>0])v using the preceding masks. The output tangent is a. For each direction e in {-1,+1}, retain the smallest positive -u/(e v) among units with u e v < 0, or infinity if empty. Only the two radii, c, T(c), a and a validity flag are needed after setup. A zero preactivation at the cache makes this conservative test abstain except at zero displacement. For nonzero displacement strictly below the directional radius, all masks persist by layer induction. At or beyond the first wall abstain. Floating-point implementations need outward error bounds on sign and boundary comparisons before calling this an exact certificate.

Write delta=q1-c, r0=T(c)-c and J for the common student Jacobian at the second evaluation. Inside the cell the fresh and cached signs are sign(r0+(a-1)delta) and sign(r0-delta) respectively. Use sign(0)=0 as in the audited absolute-loss rule. The per-example negative gradient difference is J^T(1-a)(s_f-s_c)/N. Both backward paths transmit unit derivative through their prediction argument; the cached arm implements a straight-through derivative. Identical terms suffice for an identical deterministic Adam step with shared moments and counter. A difference in one term can vanish if J=0, a=1 or other terms cancel. When r0 is nonzero, |delta| < |r0|/max(1,|a-1|) also suffices for equal signs inside the cell. This smaller bound can be computed even before q1. The exact endpoint comparison can certify equality again beyond a residual root.

## Exact fixture and limits

At c=0 take T(z)=ReLU(3-ReLU(2+z))-1/2. On (-2,1), T(z)=1/2-z. The positive wall is set by the second hidden layer. For J=N=1 the direct gradient differences are 0, -2, -4 and 0 at 1/8, 1/4, 3/8 and 3/4. At the fresh residual root the prescribed zero subgradient gives difference -2. The script compares the formula with explicit ReLU values and input derivatives on n/16 for n=-48,...,48. 42 equal, five different, 50 outside or at the cell boundary. The grid is a constructed arithmetic check. It supplies no estimate of trained-model coverage or accuracy.

## Cost, closest work and disposition

A cached forward already supplies preactivations and output. A scalar output derivative alone does not supply all hidden slopes. The dense tangent sweep costs work of the same order as one additional fixed-x reference evaluation, which can itself reuse first-layer feature contributions. The two radii permit constant-time sign comparisons for repeated hypothetical q1 values. There is one actual refresh in the audited two-step call. No timing or accuracy claim follows. Keep as a thesis diagnostic. Next test whether cheap output-only summaries could replace the sweep by constructing matching summaries with different second gradients.

Established geometry: Croce, Andriushchenko and Hein, AISTATS 2019, PMLR 89, pp. 2057-2066, activation polytope in Section 2 and distance to its boundary in Lemma 3.1. The directional intersection specializes that published construction. The residual test is a derivation for this absolute-loss graph. [Proceedings version](https://proceedings.mlr.press/v89/croce19a/croce19a.pdf).

Official code: revision 100524001d890efc08911f3cba8f45f51f4aa7d1, zero-based notebook cell 5 lines 82-109 (ReLUs), cell 13 lines 34-58 and 103 (objective and two updates). Cached-connected is our controlled intervention. See the retained cycle 15 graph audit for hashes. [Pinned notebook](https://github.com/nikitadurasov/ittt/blob/100524001d890efc08911f3cba8f45f51f4aa7d1/exps/tabular/vanilla_model.ipynb).
