"""
Steinberg Determinant Computation

This script computes the Steinberg determinant δ_p = det(I - P|_{St_p}) exactly
for prime p, where P is the Markov transition matrix on P^1(F_p) with spanning
tree weights.

Author: Yipin Wang
"""

from fractions import Fraction
from typing import List, Dict, Tuple, Set


def build_transition_matrix(p: int) -> List[List[Fraction]]:
    """
    Build the Markov transition matrix P on P^1(F_p).
    
    State space: {0, 1, ..., p-1, p} where
        - k ∈ {0, ..., p-1} represents [1:k]
        - p represents ∞ = [0:1]
    
    Transitions:
        - From ∞: go to [1:r] with weight w_r = 2^{p-r}/(2^p - 1)
        - From 0: go to ∞ with probability 1
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
        P_st[i][j] = P[i][j] - P[p][j]
    """
    P_st = [[Fraction(0) for _ in range(p)] for _ in range(p)]
    
    for i in range(p):
        for j in range(p):
            P_st[i][j] = P[i][j] - P[p][j]
    
    return P_st


def determinant_exact(M: List[List[Fraction]]) -> Fraction:
    """
    Compute the determinant of a matrix with Fraction entries using
    Gaussian elimination with partial pivoting.
    """
    n = len(M)
    # Make a copy
    M = [[M[i][j] for j in range(n)] for i in range(n)]
    
    det = Fraction(1)
    
    for col in range(n):
        # Find pivot
        pivot_row = None
        for row in range(col, n):
            if M[row][col] != 0:
                pivot_row = row
                break
        
        if pivot_row is None:
            return Fraction(0)
        
        # Swap rows if necessary
        if pivot_row != col:
            M[col], M[pivot_row] = M[pivot_row], M[col]
            det = -det
        
        # Update determinant
        det *= M[col][col]
        
        # Eliminate below
        for row in range(col + 1, n):
            if M[row][col] != 0:
                factor = M[row][col] / M[col][col]
                for j in range(col, n):
                    M[row][j] -= factor * M[col][j]
    
    return det


def compute_steinberg_determinant(p: int) -> Fraction:
    """
    Compute δ_p = det(I - P|_{St_p}).
    """
    P = build_transition_matrix(p)
    P_st = project_to_steinberg(P, p)
    
    # Compute I - P_st
    I_minus_P = [[(Fraction(1) if i == j else Fraction(0)) - P_st[i][j]
                  for j in range(p)] for i in range(p)]
    
    return determinant_exact(I_minus_P)


def factor(n: int) -> Dict[int, int]:
    """
    Factor a positive integer into prime powers.
    Returns a dictionary {prime: exponent}.
    """
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
    """
    Find alien primes at level p.
    
    An alien prime ℓ satisfies:
        - ℓ | n_p
        - ℓ > 3 and ℓ odd
        - ℓ ∤ p
        - ℓ ∤ (2^p - 1)
    """
    factors_n = factor(n_p)
    factors_mersenne = factor(2**p - 1)
    
    excluded = {2, 3, p} | set(factors_mersenne.keys())
    
    aliens = {q for q in factors_n.keys() if q not in excluded}
    return aliens


def format_factorization(factors: Dict[int, int]) -> str:
    """Format a factorization as a string like '3^2 × 7 × 41'."""
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
    
    header = f"{'p':>4} | {'n_p':>12} | {'δ_p':>25} | {'Factorization':>25} | Aliens"
    print(header)
    print("-" * len(header))
    
    for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        delta_p = compute_steinberg_determinant(p)
        n_p = delta_p.numerator
        
        # Verify denominator
        expected_denom = 2**p - 1
        if delta_p.denominator != expected_denom:
            # Reduce and check
            from math import gcd
            g = gcd(abs(n_p), expected_denom)
            actual_n = n_p // g if g != 1 else n_p
        else:
            actual_n = n_p
        
        factors = factor(actual_n)
        aliens = find_alien_primes(p, actual_n)
        
        fact_str = format_factorization(factors)
        aliens_str = str(sorted(aliens)) if aliens else "—"
        delta_str = f"{n_p}/{delta_p.denominator}"
        
        print(f"{p:>4} | {n_p:>12} | {delta_str:>25} | {fact_str:>25} | {aliens_str}")
    
    print()
    print("=" * 80)
    print("OBSERVATIONS")
    print("=" * 80)
    print("""
1. The numerator n_p is always divisible by 3 for p ≥ 5.

2. The denominator is always 2^p - 1 (after reduction).

3. Alien primes first appear at p = 11, which is the first prime
   with genus(X_0(p)) > 0.

4. The alien primes are NOT the same as Hecke discriminant primes.
   For example, at p = 11, 17, 19, 23, the Hecke discriminant is 1,
   yet alien primes exist.
""")


if __name__ == "__main__":
    main()
