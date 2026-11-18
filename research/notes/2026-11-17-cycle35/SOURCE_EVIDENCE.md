# Source evidence and scope — cycle 35

## Primary IT3 source

- Published bibliographic record checked 17 November 2026: Durasov et al., IT3: Idempotent Test-Time Training, PMLR 267 (ICML 2025), pp.14867–14883. https://proceedings.mlr.press/v267/durasov25a.html
- The supplied local PDF was read directly with the bundled pypdf reader, printed pages 3–4 (PDF pages 3–4). Its exact version is identified by SHA-256 `fe0a40e173363377c55866e1e5169135a4d1cccb96e2d3c819b395dec95f1016`; do not assume its pagination equals the published proceedings version.
- Local PDF section 2.2 / Eq.(1) defines idempotence. Section 3.1 / Eqs.(2)–(3) specifies two-input supervision and approximate consistency on the source distribution. Section 3.2 / Eqs.(4)–(5) separates the shared-weights objective from a frozen-reference objective. Section 3.3 carries student weights across online episodes; it does not supply the EMA rule studied here.
- Mathematical distinction checked independently: A^2=A means a projection; orthogonality additionally requires symmetry in the Euclidean metric. The local PDF's section 2.2 uses the word orthogonal more broadly. This audit does not attribute that wording to an unread proceedings version.
- The new construction assumes exact idempotence for every parameter in a family, much stronger than the source-training approximation. Its squared loss, stopped-output comparator and EMA dynamics are declared research models. No new claim about the official implementation is made. The earlier pinned-code audit remains in cycle-15–20 evidence.

## Closest classical caution, not a completed novelty audit

- Baird, Residual Algorithms: Reinforcement Learning with Function Approximation, ICML 1995, pp.30–37, DOI 10.1016/B978-1-55860-377-6.50013-X. Publisher record/abstract inspected: https://www.sciencedirect.com/science/article/pii/B978155860377650013X
- The abstract already contrasts unstable direct function-approximation updates with residual-gradient methods. Thus a generic claim that stopping a target derivative can alter stability is not new. Its task is Bellman learning, not this exact prediction-input architecture.
- Full author PDFs at https://leemon.com/papers/1995b.pdf and https://leemon.com/papers/1995b2.pdf failed to load in the web tool. No theorem-level comparison or claim of novelty against that paper was completed.

## Local derivation locations

- Proposal Section 3 defines q, T, r, squared discrepancy, the two backward rules, and the student-first EMA update. Section 4.2 defines H, K and the joint linearization. These were read without modification.
- Backbone subsection 'A constructed useful equilibrium and a stability distinction' uses a rotating, non-idempotent input Jacobian. Its following paragraph explicitly limits that witness under exact projection geometry.
- Cycle 12 Multivariate_Closure_Result.md section 4 contains an oblique frozen-reference example for absolute-loss differential inclusions. It does not establish the present smooth all-parameter idempotent-family/EMA calculation. The present result does not revise that nonsmooth conclusion.
- New algebra, assumptions, proof and exact model are in Idempotent_Family_Result.md; executable checks and all diagnostics are in verify_idempotent_family.py and verification.json.
