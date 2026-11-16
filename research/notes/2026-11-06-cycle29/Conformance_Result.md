# Numerical conformance and cost of the ReLU region diagnostic

6 November 2026, cycle 29. Completed the bounded implementation check specified in cycle 28.

## Outcome

All 442 accepted synthetic probes matched direct evaluation. The maximum error in the scalar negative-gradient coefficient was 1.1102230246251565e-16. The maximum absolute discrepancy in the full 1,291-parameter student gradient was 4.440892098500626e-16. There were 390 abstentions among 832 probes and zero excluded probes. Every accepted case had the same fresh and cached coefficient. Thus this screen exercises the equality and abstention branches. The changed-gradient branch is supported by the earlier exact scalar fixtures.

For a batch of 16, the median marginal time was 197.27 microseconds for region setup plus query and 133.53 microseconds for direct reference evaluation plus its input derivative, a ratio of 1.477. The Python implementation offers no demonstrated computational advantage for one query. An abstention would also require fallback. Retain this test as a thesis diagnostic and close the current one-query acceleration proposal.

## Fixed setup and verification

The script was saved and SHA256 recorded before execution as 4c3dd8833c4d3ea81b931412dbb35ed3c822f680c7860aa54d7cd01eeec27bb3. Four fixed seeds 0,1,2,3 instantiate untrained 14-30-20-10-1 fully connected ReLU networks with 1,291 parameters. The model shape and default scalar student input 50 match the audited architecture. This fixture uses PyTorch default initialization, synthetic standard-normal feature rows, batch size 16, CPU float64 and one thread. The audited call uses its own initialization, training, and data. Each reference is a frozen copy of its student.

Each cache is the initial student prediction. Probe predictions add fixed synthetic displacements 0, +/-1e-4, +/-0.01, +/-1, and in each direction half, exactly, and 1.01 times the first activation-wall distance. These offsets are imposed for conformance and do not come from an Adam step. The full student Jacobian is preserved because offsets are detached constants. The gradients checked are the fresh-connected loss, and the cached-connected loss using the straight-through expression c+(q-q.detach()), each with the same accepted-row mask and normalization by 16.

Acceptance uses directional radii and a 1e-12 numerical margin, plus residual magnitudes exceeding 1e-12. Maximum allowed numerical discrepancy was 1e-10. Any accepted discrepancy above that tolerance would reject the implementation. A zero-preactivation negative control forces abstention on a nonzero displacement. No verified rounding bound is implemented, so this is numerical conformance to an exact-arithmetic proposition.

All 128 probes of magnitude 1e-4 were accepted. Of the +/-0.01 probes, 120/128 were accepted. Of the +/-1 probes, 2/128 were accepted. All 128 half-wall probes passed. All 256 exact-wall and beyond-wall probes abstained. All 64 zero probes passed. This deliberately boundary-heavy synthetic set cannot estimate coverage on trained data.

## Cost accounting

Both timed routes exclude the common cache forward pass and all student computation. They reuse the fixed-x contribution to the first reference layer. Region setup propagates the scalar tangent through the hidden layers, constructs both radii, and compares predicted residual signs. The direct route performs a frozen-reference forward pass and PyTorch input-only reverse differentiation. It does not accumulate reference parameter gradients.

The dominant dense products in the tangent setup use 30x20+20x10+10x1=810 multiply-accumulates per row, plus masks, divisions and reductions. The direct route uses 810 downstream forward products and about 840 reverse products including the first-layer input derivative, plus scalar input additions and masks. This count excludes the common first-layer feature product. Kernel calls, reductions and Python overhead affect measured time, and fewer dense products did not produce a lower measured latency here.

Ten warmup calls precede five alternating-order blocks of 100 calls per route. All individual block means are saved in verification.json. Total experiment CPU time after imports was 0.71875 seconds, within the 60-second cap. Torch 2.9.0+cpu. The local timing is specific to this implementation and batch size, and is separate from an end-to-end IT3 measurement.

## Source and claim status

Rechecked the retained official notebook extraction, SHA256 ead7cade66b9df7fa7e90d4c66d986b75cd847d23473d4ea86f80e3737ee46d2. Revision 100524001d890efc08911f3cba8f45f51f4aa7d1, zero-based cell 5 lines 82-109 defines the architecture, cell 13 lines 34-38 the connected loss, lines 41-58 the reset/update loop, and line 103 its two-step call. The fixture reuses the model layout from cycle 16 conformance.py, avoiding its executable top-level experiment.
https://github.com/nikitadurasov/ittt/blob/100524001d890efc08911f3cba8f45f51f4aa7d1/exps/tabular/vanilla_model.ipynb

Cycle 27 records the exact local-affine proof and its established geometry from Croce et al., AISTATS 2019 Section 2 and Lemma 3.1. Cycle 28 records why output summaries cannot replace the missing region information. The present result checks numerical code and marginal cost. It establishes no accuracy, retention, recovery or novelty claim.

## Next substantive question

The diagnostic sequence is complete. The thesis's empirical limitation is the single trained checkpoint. Next perform a bounded feasibility audit for replication across independent initializations using the locked cycle-18 training recipe and original split. Read the existing training protocol before proposing any run, predeclare seed choices and compute budget, preserve the original failed accuracy gate, and treat already observed evaluation targets as retrospective evidence. The next decision is whether an affordable unchanged-recipe replication can determine how sensitive the reported harm is to initialization.
