# October 15, 2026: manuscript-result synchronization

Based on the existing GitHub main branch; earlier remote commits are preserved.

- Unify all five primary dynamic outcomes as Low/High-only Condition × Progress, diagonal participant intercept/condition/progress random effects; retire the separate Single-inclusive gap fit. Apply BH across five primary interactions.
- Use Intersyllabic-gap in Figures 2–4 while preserving input field names and the current manuscript panel layout.
- Correct Figure 4 annotation to Dual–Single without reversing effect values; preserve remote age-scaling and convergence guards.
- Export both DDK Regularity endpoints in Table 3 (+3.03% at Trial 1, +6.11% at Trial 10). Missing endpoints must fail verification.
- Retain historical targets, add independently locked current Table 3 targets, and expand the verifier to 244 checks.
- Retain the current age/CMMS model results pending review of covariance identifiability and condition-level covariate moderation; explain these limitations in both READMEs.
- Describe the two sensitivity subsets independently and restrict the role of manual annotation to supplementary measurement agreement.
- Update the manuscript title, recording/stimulus timing explanation and public-file exclusions. Do not publish participant-level data or local supplementary experiments.

Validation: Figures 2–4 and Table 3 rebuilt from local controlled inputs and existing result tables; source/target aggregate checks performed in env_codex. A missing-DDK-Regularity-endpoint regression check must fail. The complete research pipeline was not rerun, and no new age/CMMS statistical conclusion is claimed. Historical numerical targets and reference images are unchanged.
