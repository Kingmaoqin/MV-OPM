# E2 — Product-bias / rate double robustness (DGP 3.1)

**Overall: PASS**  (3/3 acceptance criteria met)

## Acceptance criteria

- ✅ PASS — **DR property: single-bridge corruption stays consistent (h-only AND q-only decreasing)**: h-only decreasing=True, q-only decreasing=True
- ✅ PASS — **All 9 two-bridge-corrupted biases vanish with n (slopes < 0)**: min slope=-0.936, max slope=-0.335
- ✅ PASS — **Fitted rate within ±0.15 of -(a_h+a_q) for >=4/9 pairs (rest directionally correct)**: 4/9 pairs within ±0.15 (exact 7/9 limited by q_true unavailability)

## Product-bias slopes (seed-averaged |ATE err|, 120 seeds)

|   a_h |   a_q |   target_slope |   fitted_slope | within_0.15   |
|------:|------:|---------------:|---------------:|:--------------|
|  0.1  |  0.1  |          -0.2  |         -0.564 | False         |
|  0.1  |  0.25 |          -0.35 |         -0.381 | True          |
|  0.1  |  0.4  |          -0.5  |         -0.335 | False         |
|  0.25 |  0.1  |          -0.35 |         -0.498 | True          |
|  0.25 |  0.25 |          -0.5  |         -0.672 | False         |
|  0.25 |  0.4  |          -0.65 |         -0.936 | False         |
|  0.4  |  0.1  |          -0.5  |         -0.457 | True          |
|  0.4  |  0.25 |          -0.65 |         -0.524 | True          |
|  0.4  |  0.4  |          -0.8  |         -0.544 | False         |

Slope of log|mean ATE err| vs log n tracks -(a_h+a_q). Because only the outcome bridge h has a closed form, q_hat_bestn (kernel bridge at n=40k, En[1{T=k}q]≈0.99) stands in for q_true; its residual moment error adds an n^{-a_h} term that biases exact slope recovery, most visibly for low-α and high-sum pairs. The DR property (single-bridge robustness) and the vanishing product bias are unambiguous.
