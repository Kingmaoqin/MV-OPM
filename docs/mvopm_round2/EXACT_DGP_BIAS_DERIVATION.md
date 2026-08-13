# Aligned exact-DGP product-bias derivation

Consider treated arm 1 and keep the reference arm 0 exactly correct. Write the exact treated-arm
bridges as h and q and define `g(X)=tanh(X_1)`. Round 2 uses

`h_r = h + r_h g`, and `q_r = q(1+r_q g)`.

The treated-arm proximal pseudo-outcome is

`phi(h_r,q_r) = I(T=1) q_r (Y-h_r) + h_r`.

Subtracting the exact-bridge pseudo-outcome and expanding gives

`r_q g Iq(Y-h) + r_h g[1-Iq(1+r_q g)]`.

The first term has mean zero because the outcome bridge implies
`E[Y-h(W,X) | V,X,T=1]=0`. For the second term, the exact treatment bridge implies
`E[I(T=1)q(V,X) | W,X]=1`. Because g is a function of X,

`E{r_h g[1-Iq(1+r_q g)]} = r_h E{g[1-(1+r_q g)]}`

`= -r_h r_q E[g(X)^2]`.

Thus the treated-arm population mean bias, and hence the arm-1-versus-0 contrast bias when the
reference arm remains correct, is exactly

`Bias_ATE(r_h,r_q) = -r_h r_q E[tanh(X_1)^2]`.

This establishes the two single-side checks: at `r_q=0` or `r_h=0`, population bias is zero. It
also explains why the old sin/cos perturbations nearly canceled: their cross-product was
orthogonal/zero-mean under independent Gaussian X coordinates.

No outcome-scale normalization is used: in this DGP the outcome-noise standard deviation is one,
so r_h is measured directly in outcome units. r_q is a dimensionless relative perturbation.

For `X_1 ~ N(0,1)`, independent 256-node Gauss-Hermite quadrature gives

`E[tanh(X_1)^2] = 0.3942944903978411`.

This value is computed deterministically and is never estimated from a causal Monte Carlo sample.
The predicted biases at the corruption grid are stored alongside every A2 raw row.

Finally, `|g|<1` and `r_q<=0.40`, so `1+r_q g` lies strictly between 0.60 and 1.40. The exact q
bridge is positive in the frozen finite-proxy DGP; therefore the perturbed q remains positive.
The implementation checks and records the minimum multiplier and minimum q rather than repairing
failures by clipping.
