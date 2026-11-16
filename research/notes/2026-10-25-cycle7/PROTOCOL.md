# Frozen prediction-input control: protocol before results

Saved 24 October 2026, cycle 6. No control trajectories have been run. This is the next single experiment, not a new adaptation method.

## Hypothesis and exact intervention

Hypothesis: the changing student prediction fed into the reference materially drives the previously observed stationary risk deterioration, beyond teacher-weight averaging alone.

For each existing cycle-3 checkpoint p0, cache c(x)=f_p0(x,0) on the same 64 adaptation inputs once. The control residual is f_phi(x,c(x))-f_theta(x,0), with c fixed throughout. It minimizes the same squared loss through the student output only. In contrast, the original residual recomputes f_phi(x,f_theta(x,0))-f_theta(x,0) at every step. The original stopped-output rule is the primary comparison: both arms have identical initial targets AND velocities. Simply detaching a recomputed prediction input is not this intervention.

## Fixed budget and information

- Reuse all five checkpoints, source code/data hashes and split from cycles 3-5. Primary seeds 0,2,3,4; seed 1 remains sensitivity-only. No retraining or extra seeds.
- Use the identical repeated 64-input adaptation batch; the remaining 192 inputs and labels are evaluator-only. The labels never enter a gradient, update, cache, hyperparameter decision, stopping rule, or method choice within the run.
- Six configurations per seed: original stopped-output, original full input-gradient, and frozen-input control; each with fixed reference or EMA tau=1. No tau/horizon sweep. No-adaptation is evaluated once per seed. Full-gradient versus control is secondary because its initial velocity differs.
- Student-first GD/EMA, eta=.02, 1000 steps (T=20), alpha=1/(1+eta). Repeat all six base runs with eta=.01 for 2000 steps at the same horizon. Budget 90,000 student updates, 30 main and 30 numerical-check trajectories; target <=5 CPU minutes, hard cap 10 CPU minutes. No tangent trajectories or additional perturbations are required. If the cap is reached, save incomplete status and do not select completed seeds as a new primary set.
- Same 114-parameter architecture, 82 non-fixed student coordinates, two parameter copies, no replay or optimizer state beyond GD. Allocate/cache the same 64x2 float64 predictions in every condition (128 scalars, 1024 bytes); only the control reads them during updates. This cache is additional retained test-input-associated information, declared explicitly. No training labels or extra source model are stored. A future fresh-sample stream would require a different protocol.

## Checks before interpreting results

1. Verify model/data/checkpoint hashes against cycle 4. Check control loss gradient by central differences, maximum absolute error <=1e-7 at all initial checkpoints.
2. Check initial target and stopped/control velocity equality to <=1e-12. Independently differentiate both full fields and verify S_stop-S_control=E[J^T A J] and T_stop-T_control=0 to maximum absolute error <=1e-7. Hold c fixed during every derivative.
3. Reproduce the original arms' saved cycle-4 base endpoint NMSE within relative 1e-6 or absolute 1e-8. A mismatch must be diagnosed, not silently treated as a new result.
4. Halving the step must change final NMSE by <=5%, using denominator max(abs(fine-step NMSE),.01). If this fails for any primary paired arm, mark the overall risk comparison numerically unresolved; report both runs and do not tune the step in the same confirmatory run.

## Outcomes and decision fixed in advance

Record initial/final NMSE on the 192-point evaluator using cycle-4 label-variance normalization; save every 20 main steps and 40 fine steps, including discrepancy, reference lag and parameter states. Also measure RMS change of the actual reference targets on the adaptation batch from their initial values. Frozen-weights/frozen-input targets must remain constant to <=1e-12; frozen weights alone need not make original targets constant. This checks the implementation, not label correctness.

Let R0 be the seed's initial NMSE, Re the original stopped-output/EMA final NMSE, and Rc the frozen-input/EMA final NMSE. A primary seed supports the hypothesis only if Re >= 2 R0, Re-R0 >= .01, and max(Rc-R0,0) <= .25 (Re-R0). This asks whether freezing the input removes at least 75% of the excess terminal error when the original deterioration is material. The primary gate passes only if at least three of four primary seeds satisfy all inequalities in both step sizes and all numerical checks pass. Report individual seeds and full trajectories, not just gate counts.

If the gate fails, reject the proposed dominant contribution of the changing prediction input at this horizon/model; retain any smaller or mixed effect without redefining success. A pass establishes a controlled synthetic observation, not a universal guarantee, useful adaptation under drift, retention, recovery, or novelty. Fixed-reference controls, full-gradient arms, reference-target drift, discrepancy and seed 1 are descriptive secondary results, not replacements for a failed primary gate. Do not introduce a new stabilizer or optimize the cache after seeing results.
