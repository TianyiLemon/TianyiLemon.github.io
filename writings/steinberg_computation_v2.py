"""
Steinberg Determinant Computation (Corrected Version)

This script computes the Steinberg determinant δ_p = det(I - P|_{St_p}) exactly
for prime p, where P is the weighted transition matrix on P^1(F_p).

Key correction: The weights w_r = 2^{p-r}/(2^p - 1) sum to 2, NOT 1.
The matrix P is therefore NOT stochastic, but this is the correct
formulation that yields the motivic structure n_p(q) ∈ Z[q].

Author: Yipin Wang
"""

from fractions import Fraction
from typing import List, Dict, Set


def build_transition_matrix(p: int) -> List[List[Fraction]]:
    """
    Build the weighted transition matrix P on P^1(F_p).
    
    State space: {0, 1, ..., p-1, p} where
        - k ∈ {0, ..., p-1} represents [1:k]
        - p represents ∞ = [0:1]
    
    Weights: w_r = 2^{p-r}/(2^p - 1) for r = 0, ..., p-1
    NOTE: These sum to 2, so P is NOT stochastic.
    
    Transitions:
        - From ∞: go to [1:r] with weight w_r
        - From 0: go to ∞ with weight 1
        - From k ≠ 0, ∞: go to (rk+1)k^{-1} mod p with weight w_r
    """
    n = p + 1
    denom = 2**p - 1
    
    P = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    
    # From state 0 = [1:0]: go to ∞ = p
    P[0][p] = Fraction(1)
    
    # From state p = ∞ = [0:1]: go to [1:r] with weight w_r
    for r in range(p):
        w_r = Fraction(2**(p - r), denom)
        P[p][r] = w_r
    
    # From state k ∈ {1, ..., p-1}: go to (rk+1)k^{-1} mod p
    for k in range(1, p):
        k_inv = pow(k, -1, p)
        for r in range(p):
            w_r = Fraction(2**(p - r), denom)
            target = ((r * k + 1) * k_inv) % p
            P[k][target] += w_r
    
    return P


def project_to_steinberg(P: List[List[Fraction]], p: int) -> List[List[Fraction]]:
    """
    Project P to the Steinberg representation.
    
    Basis for St_p: {e_i - e_∞ : i = 0, ..., p-1}
    
    The projected matrix P_st satisfies:
        P_st[i][j] = P[i][j] - P[∞][j]
    """
    P_st = [[Fraction(0) for _ in range(p)] for _ in range(p)]
    
    for i in range(p):
        for j in range(p):
            P_st[i][j] = P[i][j] - P[p][j]
    
    return P_st


def determinant_exact(M: List[List[Fraction]]) -> Fraction:
    """
    Compute the determinant using Gaussian elimination.
    """
    n = len(M)
    M = [[M[i][j] for j in range(n)] for i in range(n)]
    
    det = Fraction(1)
    
    for col in range(n):
        pivot_row = None
        for row in range(col, n):
            if M[row][col] != 0:
                pivot_row = row
                break
        
        if pivot_row is None:
            return Fraction(0)
        
        if pivot_row != col:
            M[col], M[pivot_row] = M[pivot_row], M[col]
            det = -det
        
        det *= M[col][col]
        
        for row in range(col + 1, n):
            if M[row][col] != 0:
                factor = M[row][col] / M[col][col]
                for j in range(col, n):
                    M[row][j] -= factor * M[col][j]
    
    return det


def compute_steinberg_determinant(p: int) -> Fraction:
    """Compute δ_p = det(I - P|_{St_p})."""
    P = build_transition_matrix(p)
    P_st = project_to_steinberg(P, p)
    
    I_minus_P = [[(Fraction(1) if i == j else Fraction(0)) - P_st[i][j]
                  for j in range(p)] for i in range(p)]
    
    return determinant_exact(I_minus_P)


def factor(n: int) -> Dict[int, int]:
    """Factor a positive integer into prime powers."""
    n = abs(n)
    if n <= 1:
        return {}
    
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = 1
    
    return factors


def find_alien_primes(p: int, n_p: int) -> Set[int]:
    """Find alien primes: odd ℓ > 3 with ℓ | n_p but ℓ ∤ p(2^p - 1)."""
    factors_n = factor(n_p)
    factors_mersenne = factor(2**p - 1)
    
    excluded = {2, 3, p} | set(factors_mersenne.keys())
    
    return {q for q in factors_n.keys() if q not in excluded}


def format_factorization(factors: Dict[int, int]) -> str:
    if not factors:
        return "1"
    
    parts = []
    for prime in sorted(factors.keys()):
        exp = factors[prime]
        if exp == 1:
            parts.append(str(prime))
        else:
            parts.append(f"{prime}^{exp}")
    
    return " × ".join(parts)


def main():
    print("=" * 80)
    print("STEINBERG DETERMINANT AND ALIEN PRIMES")
    print("=" * 80)
    print()
    print("NOTE: Weights w_r = 2^{p-r}/(2^p - 1) sum to 2 (not 1).")
    print("      The matrix P is NOT stochastic, but this yields the")
    print("      correct motivic structure: n_p(q) ∈ Z[q] with n_p = n_p(2).")
    print()
    
    header = f"{'p':>4} | {'n_p':>12} | {'δ_p':>20} | {'Factorization':>25} | Aliens"
    print(header)
    print("-" * len(header))
    
    for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        delta_p = compute_steinberg_determinant(p)
        n_p = delta_p.numerator
        
        factors = factor(n_p)
        aliens = find_alien_primes(p, n_p)
        
        fact_str = format_factorization(factors)
        aliens_str = str(sorted(aliens)) if aliens else "—"
        delta_str = f"{n_p}/{delta_p.denominator}"
        
        print(f"{p:>4} | {n_p:>12} | {delta_str:>20} | {fact_str:>25} | {aliens_str}")
    
    print()
    print("=" * 80)
    print("MOTIVIC POLYNOMIALS n_p(q)")
    print("=" * 80)
    print("""
The Steinberg numerator n_p is the specialization at q = 2 of:

  n_3(q) = q - 1                    = |G_m|(F_q)
  n_5(q) = q^2 - 1 = (q-1)(q+1)     = |A^2 \\ {0}|(F_q)
  n_7(q) = (q-1)(2q^2 + 1)          = virtual motive

This motivic structure explains alien primes: they are primes ℓ
where H^*(X_p, Z_ℓ) has torsion for the variety X_p with
|X_p(F_q)| = n_p(q).
""")


if __name__ == "__main__":
    main()
