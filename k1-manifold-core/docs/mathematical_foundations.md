# Mathematical Foundations

The core package follows the paper's layer separation:

1. Information time motivates a signed leading quadratic form `G`.
2. Axioms R, E, T plus nondegeneracy force `Sig(G) = (1, 1)` in two dimensions.
3. `K(x) = x.T @ G @ x = 1` is added as a point-level self-consistency condition.
4. Law II gives `xdot = (J_G - D) grad V`, where `J_G = alpha G^-1 J`.
5. Law III selects `D = d_c I` in the isotropic critical-damping reading.

The code keeps external thermodynamic assumptions explicit. In particular, `T_eff = T_tol` is represented as a checkable identification, not as a theorem.
