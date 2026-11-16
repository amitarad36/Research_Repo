# Output summaries leave the second gradient unidentified

6 November 2026, cycle 28. Checked construction and information limit. This completes the analytical follow-up to cycle 27. No new training, checkpoint evaluation, labels or external compute.

## Precisely stated question and answer

Can a rule receiving only c, T(c), T'(c), the common second student prediction q1, its Jacobian, the optimizer state and even the fresh endpoint value T(q1) always decide whether refreshing changes the connected second gradient? For scalar ReLU frozen references the answer is no. A sound rule must abstain on the matching summaries constructed here or obtain additional information. This rules out a universally complete decision using this summary. It leaves room for conservative tests on other summaries and for methods inspecting weights, activation geometry or additional derivatives.

## Construction and algebraic verification

For any 0<d<1, c=0, q1=d and J=N=1, define
T_k(z)=1+k ReLU(z-d/3)-2k ReLU(z-2d/3), k=0,1,-2.
At zero both hidden units are strictly inactive. At d both are strictly active. The three functions have identical cached output 1, cached derivative 0 and refreshed output 1. Their refreshed derivatives are -k. The two residual signs are positive because 1-d>0. Therefore the cached-connected negative gradient is 1 in all three cases, and the fresh-connected negative gradient is 1+k. Its values 1,2,-1 include equality, doubling and reversal. All endpoint derivatives exist. The piecewise slopes are 0,k,-k, so a common known Lipschitz upper bound 2 does not resolve the ambiguity.

The networks agree on a neighborhood of the cache. The first nontrivial activation wall is at d/3 for k=1 or -2, making cycle 27 abstain correctly. For k=0 the same syntactic hidden walls have no output effect, illustrating conservative abstention. As d can be arbitrarily small, these summaries cannot establish a uniform positive safe radius at (c,T(c),T'(c))=(0,1,0). The construction varies the network with d. Each fixed network remains affine on its own local neighborhood.

The student and frozen reference are independently specified in this counterexample class. Taking q_theta=theta with theta=0 initially yields first negative gradient 1 for all three references. A fresh scalar Adam optimizer produces the same positive first displacement eta/(1+epsilon), so the counterexample can use this realized d and share optimizer history. This only checks consistency of the common first update. It does not embed the student and reference as tied initial copies of the original 1,291-parameter checkpoint. A reversing second gradient does not itself establish a reversing Adam step, because Adam retains first moments. Accuracy is unexamined.

## Reproducible checks

verify_summary.py uses exact fractions for four displacements and three coefficients, giving 12 cases. One displacement is 10000/100000001, the fresh-Adam value at eta=1e-4 and epsilon=1e-8. All summary collisions and field formulas pass. Twenty-four independent central finite-difference calculations stay inside the endpoint cells and match the analytic slopes exactly. This is a constructed check, with no trained-model coverage estimate.

## Evidence and disposition

The frozen-reference connected formula and implementation locations are preserved in cycle 15 Graph_Audit_Result.md and cycle 27 Region_Result.md. The new calculation applies that formula. Prior geometry is the activation-polytope construction in Section 2 and Lemma 3.1 of Croce, Andriushchenko and Hein, AISTATS 2019, PMLR 89, pp. 2057-2066. Inspected the proceedings PDF again on 6 November 2026. This primary source establishes the need to remain within a certified activation region for its local affine expression.
https://proceedings.mlr.press/v89/croce19a/croce19a.pdf

Official IT3 source remains pinned at 100524001d890efc08911f3cba8f45f51f4aa7d1. Notebook cell 5 lines 82-109 specifies ReLUs. Cell 13 lines 34-58 and 103 specifies the connected loss and two updates. The cached-connected graph is a controlled intervention in our analysis.
https://github.com/nikitadurasov/ittt/blob/100524001d890efc08911f3cba8f45f51f4aa7d1/exps/tabular/vanilla_model.ipynb

Novelty remains provisional. Keep the construction as an information-budget limitation in the thesis. The mathematical diagnostic package is complete at this scope. Before suggesting an implementation advantage, the next bounded task is a data-free conformance and operation-cost check of the hidden-slope implementation on the audited ReLU architecture, with synthetic inputs, deterministic seeds, zero training and a 60-CPU-second cap. Compare direct fresh evaluation with the region method, count certified/abstained cases and compare gradients only on certified cases. Any mismatch rejects the implementation. Report timings as local measurements and do not treat synthetic coverage as trained-model evidence. Restrict any floating-point claim to numerical agreement unless verified error bounds are implemented.
