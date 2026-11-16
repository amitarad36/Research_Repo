# Next decision: isolate the forward input path in the audited backward graph

One bounded static/analytic audit; no new training, adaptation trajectory,
labels, packages or benchmark. Reuse the exact saved official notebook
revision 100524001d890efc08911f3cba8f45f51f4aa7d1, selected tabular cell 13,
and existing code audit. Verify hashes and exact loss/optimizer/reset rules.

Question: does the existing stopped-versus-cached diagnostic isolate a
mechanism applicable to the implementation that keeps the input gradient?
Separate two factors explicitly: reference-input forward value (current q
or cached initial c) and its derivative with respect to q (identity or zero).
Audit four computational graphs. A candidate cached-value/identity-backward
graph can be expressed as c+q-stop_gradient(q), but its exact backward
meaning and relationship to the original code must be derived, not assumed.
Its forward value is c and input derivative is I; it is not an ordinary
gradient of the forward loss. Do not call it the official method.

Derive the four student directions for squared and mean-absolute discrepancy
with the correct normalization. At matched initialization c=q0, identify
which forward values, gradients and first Adam updates must agree; distinguish
this from later sign/activation/optimizer-history changes. Declare cache
scalars and reset lifetime. Frozen parameters do not imply detached inputs.

Deliver a compact graph/claim table, a scoped equality or counterexample
with algebraic verification if needed (max 10 CPU seconds), and a decision:
either one clear implementation-relevant causal question with a prospective
test plan, or a reason to stop this intervention branch. No automatic
experiment or publication claim. This is an implementation-relevance gate,
not a rescue of the failed certificate or absolute-loss numerical gate.
