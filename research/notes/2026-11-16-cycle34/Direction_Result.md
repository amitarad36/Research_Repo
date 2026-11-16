# First-step Adam history changes direction
16 November 2026, cycle 34. Reproduced Adam algebra and new checked diagnostics on existing trajectories. No novelty claim, new training, model calls, optimizer steps, raw data or target access.

## Question and answer
Can the first update with carried optimizer state be reproduced by a nonnegative scalar change of the fresh Adam update? In parameter space the answer is no, at the registered tolerance, for all 76 batches after the first batch in each seed. The four initial batches agree within rounding. The current gradient and starting weights are common, so this separates a direction effect of optimizer history from a learning-rate-only explanation at those saved states. It does not identify a risk benefit or a recovery mechanism.

| Seed | Noninitial batches with parameter discrepancy | Median relative parameter residual | Median relative linearized-output residual | Median relative actual-output residual |
|---|---:|---:|---:|---:|
| 0 | 19/19 | 70.531% | 1.296% | 1.621% |
| 1 | 19/19 | 52.497% | 3.962% | 3.967% |
| 2 | 19/19 | 90.498% | 16.186% | 16.179% |
| 3 | 19/19 | 55.004% | 1.654% | 1.463% |

Every linearized-output and stored-output comparison also exceeds the registered relative threshold. The displayed summaries exclude the first batch solely because it is the explicitly reported zero-history equality control; all 80 records are retained. The best fit is a ray, a>=0, so a residual of 1 can include an opposing direction whose best permitted scale is zero. Do not interpret it as an orthogonal angle or a percentage of error explained.

## Exact algebra and its scope
Let g be the common current gradient, m,v the incoming moments, n the next logical update count. All coordinates share n. No weight decay or AMSGrad. beta1=.9, beta2=.999, eta=1e-4, epsilon=1e-8.

V_H=beta2*v+(1-beta2)*g^2; V_Z=(1-beta2)*g^2.
Define diagonal multipliers elementwise
D_H=-eta/[(1-beta1^n)*(sqrt(V_H/(1-beta2^n))+epsilon)],
D_Z=-eta/[(1-beta1^n)*(sqrt(V_Z/(1-beta2^n))+epsilon)].
The parameter displacements are
u_H=D_H*[beta1*m+(1-beta1)*g],
u_Z=D_Z*[(1-beta1)*g],
u_F=-eta*g/(|g|+epsilon).

The exact same-counter identity is
u_H-u_Z = D_H*(beta1*m) + (D_H-D_Z)*((1-beta1)*g).
First term retains the historical denominator and adds the old first moment. The second changes the denominator for the current-gradient numerator. This is an explicitly ordered attribution. Reversing the order changes the component split; both terms may cancel. It is neither a unique causal allocation between moments nor an additive allocation of prediction error. Left multiplication by the saved initial student Jacobian J preserves the identity.

For zero moments define a_n=(1-beta1)/(1-beta1^n), b_n=sqrt[(1-beta2)/(1-beta2^n)], c_n=a_n/b_n. If epsilon=0, on coordinates with g nonzero u_Z=c_n*u_F. With the actual positive epsilon, the exact correction is
u_Z-c_n*u_F = -eta*a_n*g*epsilon*(b_n-1) /
[b_n*(b_n*|g|+epsilon)*(|g|+epsilon)].
It is defined as zero when g=0 with positive epsilon. Thus counter-only effects with cleared moments are not exactly scalar in the implemented optimizer. The maximum relative deviation from c_n*u_F is .2952%, .3021%, .0371%, .4031% for seeds 0/1/2/3. Best-scalar parameter residuals are also small (maximum .403%). This contrasts with the much larger carried-history residuals; no epsilon value was retuned.

## Measurement
For h,f, solve min_{a>=0} ||h-a*f||_2 by a=max(0,h.f/(f.f)). A direction event requires residual >1e-6*max(||h||,1e-12). Degenerate reference threshold f.f<=1e-28 was fixed before execution. Parameter fits use analytic steps, independently checked against saved parameter subtraction. Linear output fits use J*u. Actual output fits use saved q_after_first-q_before; they are a separate diagnostic.

Critical distinction: fitting the two stored output displacement vectors does NOT evaluate the nonlinear model at theta+a*u_F. We have not shown that every rescaled fresh parameter step fails to match the carried output. Nor do the 2-output probes characterize behavior elsewhere. Parameter-space direction differences depend on the chosen parameterization. Inactive-gradient coordinates can move through stored moments; their count and nonlinear remainder norms are retained for every batch.

This is one first update conditional on archived two-update histories. No multi-step fit, deployed reset-policy comparison, label alignment or risk inference is claimed. Previously observed same-cohort accuracy results remain as in cycle 33. The same-counter moment-zero arm also differs directionally from carried state in all 76 noninitial batches.

## Validation and registration
PROTOCOL.md and attribute.py hash-locked before diagnostics:
registration SHA256 a5a826745a67f31d0628eefbc65e0f0609dcd3bfe29d7f312b06f06743adcc57.
Source manifests for cycles 33 and 31 verified, then trace/script/protocol hashes checked at execution. Cycle 19 supplies seed-zero J, cycle 31 seeds 1/2/3; common gradients and baseline predictions match archived values exactly.

All 80 records pass. Maximum absolute formula-versus-saved step error 5.54e-17; decomposition error 2.72e-20; projected decomposition 1.39e-17; epsilon-correction identity 4.07e-20. Scalar math.fsum agrees with the vector dot product within 8.48e-22. Fit scale based on saved subtraction agrees within 2.23e-14, residual norm within 3.11e-17. No nonfinite values. CPU 1.015625 seconds after imports, cap 10 seconds. Complete data: attribution_results.json.

## Primary sources
- Kingma and Ba, Adam, arXiv:1412.6980v9, Algorithm 1 p.2, as previously audited in cycle23: https://arxiv.org/pdf/1412.6980v9 . The HTML route was checked this cycle and only supplied a PDF link; no newly readable HTML paper text was available.
- PyTorch 2.9 Adam official documentation, algorithm lines 88-97 in the fetched page and eps definition line111, checked 16 November 2026: https://docs.pytorch.org/docs/2.9/generated/torch.optim.Adam.html . This directly specifies the two moment recurrences, bias correction and denominator used above.
- IT3 tabular notebook commit100524001d890efc08911f3cba8f45f51f4aa7d1, cell13 lines34-58 loss/reset/updates,61-64 and90-95 optimizer lifetime,103 two updates: https://github.com/nikitadurasov/ittt/blob/100524001d890efc08911f3cba8f45f51f4aa7d1/exps/tabular/vanilla_model.ipynb . Local reproduction trace indexing verified against cycle33/intervene.py; no remote revision change was assumed.

## Submission decision and next step
This answers the saved-state direction-versus-size question. It strengthens the mechanism chapter; it does not establish a new optimizer or a paper claim. Given the user's approaching submission, the next priority is to consolidate the existing proposal around three objectives: identify the IT3 gradient/input mechanisms, attribute optimizer-state effects with matched controls, and test the resulting explanations independently while stating what information accuracy guarantees require. Preserve detailed older branches as supporting research material. Retention, EMA and recurring-domain recovery remain conditional extensions.

A prospective independent validation should be specified before new data are scored, with fixed seeds and matched steps/learning rates, reporting failures and both parameter and prediction consequences. A finite-step prediction claim needs actual forward evaluation of the scaled baseline, not the output-vector fit alone. Page limit/template and deadline were requested in chat but remain unknown; do not present the current long technical record as submission-ready.
