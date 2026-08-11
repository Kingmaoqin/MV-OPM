"""OPM v2 — Orthogonal Proximal Moments.

A multi-treatment proximal doubly-robust CATE learner with an optional
diffusion-distillation module for distributional counterfactuals.

Design principles (see the agent spec / DECISIONS.md):
  * Sequential pipeline, never joint training (Stages 1->2->3, each frozen for the next).
  * Exactly one tunable hyperparameter in the core method: ``lambda_distill``.
  * Cross-fitting mandatory for all nuisance functions; pseudo-outcomes are out-of-fold.
  * Honest degradation to a plain multi-treatment DR-learner when proxies are absent.
"""

__version__ = "2.0.0"
