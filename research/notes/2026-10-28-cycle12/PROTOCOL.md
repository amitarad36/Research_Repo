# Next bounded transfer audit: the correct multivariate closure

Saved 28 October 2026 after the scalar nonsmooth baseline was completed. This is a theoretical follow-up, not an empirical confirmation protocol. No claim of a vector theorem is made yet.

Question: when does replacing each zero residual sign by an independently chosen value in [-1,1] equal the Filippov closure of the actual stopped-output student/reference field? If it is only an outer approximation, can it introduce spurious velocities or solutions that invalidate a claimed Lyapunov or well-posedness result?

Use smooth finite-batch maps q_theta, T_phi and a specified forward input (current prediction or cached prediction), with the true negative-gradient field for each backward rule. Treat a fixed reference as a parameter and a moving reference as a state; do not conflate perturbations of a fixed parameter with neighborhoods of the adaptive state. Preserve all mean factors and shared sign choices across vector components. Start from the exact neighborhood/convex-closure definition of Filippov, not an assumed independent box.

Deliver one precise sufficient condition for equality, or a counterexample to an unrestricted equality claim, ideally both if they share one proof. Examine locally reachable joint sign patterns at simultaneous zero residuals and explicitly state a rank/transversality assumption if needed. Every example must arise from one declared dual-input forward model; compare the full input-gradient rule only when its algebra justifies it. Distinguish an outer differential inclusion from a faithful limiting dynamics and do not infer discrete convergence merely from closure equality.

Bounded work: one cycle, primarily proof and a small exact symbolic/rational check if useful, <=30 CPU seconds. Reuse the saved mathematical sources before adding only the specific primary source needed for a disputed claim. No neural retraining, extra seeds, adaptation trajectories, labels, smoothing or eta sweep. Cycle 10's endpoints and unresolved numerical classification remain unchanged.

Pass: a proved, correctly scoped equality criterion or a realizable counterexample with the true closure computed. Fail: if neither is established, record the exact gap and do not present an independent-sign box as the model. Reassess the first-paper direction explicitly: a standard closure observation alone is still groundwork, and a credible paper needs a distinct predictive consequence plus implementation relevance. Do not turn this into an open-ended sequence of toy examples or a broad new reading list.
