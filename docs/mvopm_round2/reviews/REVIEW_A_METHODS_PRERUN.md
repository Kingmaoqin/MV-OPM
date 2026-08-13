# Reviewer A — methods/statistics pre-run audit

**Verdict: PASS after fixes.** No Round-2 final task had been launched.

The isolated reviewer audited the exact aligned DGP, formal moment tests, union-null and
Bonferroni screening, OOF uncertainty target, nested selection/refit, convergence rule, and CATE
relative-risk boundary. Initial major findings were fixed before final execution:

1. The CATE analysis now derives the conditions needed for a proximal signal to replace the
   unknown CATE, implements only the pairwise functional, and labels it secondary/experimental.
2. MSES headline metrics now come from outer-test predictions after inner selection and refit.
3. The convergence audit ran S1/S2/nonlinear/HAMD × eight reserved development seeds and froze
   300 epochs using observed-data criteria only.
4. All adaptive comparators now select within the identical inner-OOF population and refit/predict
   on the same outer splits as MSES.
5. Development/final bootstrap draws are 499/999.

The reviewer independently verified the exact bias sign, reference handling, q positivity,
studentization, centered Gaussian multiplier bootstrap, plus-one p-value, isolated bootstrap
streams, `p_DR=max(p_h,p_q)`, both-sides rejection, and mean contrast-variance criterion.

The full pre-run invariant suite passed 27/27. CATE relative-risk confidence intervals must not be
described as pooled cross-fit-valid without fold-aware inference.
