# Written by Claude Code (claude-sonnet-4-6)

"""Generate hardcoded C det functions for dimensions 1..N via cofactor expansion.

Usage: python3 hardcoded_dets_in_c.py [--max <int>] [--out <file>]
  --max <int>   Maximum dimension (default: 6)
  --out <file>  Output file (default: stdout)
"""

import sys
from itertools import permutations


def gen_terms(n):
    """Return list of (sign, [col indices]) for the Leibniz expansion of an nxn det."""
    terms = []
    for p in permutations(range(n)):
        # compute sign of permutation
        sign = 1
        visited = [False] * n
        for i in range(n):
            if not visited[i]:
                j = i
                cycle_len = 0
                while not visited[j]:
                    visited[j] = True
                    j = p[j]
                    cycle_len += 1
                if cycle_len % 2 == 0:
                    sign *= -1
        terms.append((sign, list(p)))
    return terms


def term_to_c(sign, cols, n):
    """Convert a single term to a C expression like '+M[0*n+2]*M[1*n+0]*...'"""
    factors = [f"M[{row}*{n}+{col}]" for row, col in enumerate(cols)]
    product = "*".join(factors)
    if sign == 1:
        return f"+ {product}"
    else:
        return f"- {product}"


def gen_function(n):
    terms = gen_terms(n)
    lines = []
    lines.append(f"static int det{n}(int *M) {{")
    lines.append(f"    return")
    term_strs = [f"        {term_to_c(s, cols, n)}" for s, cols in terms]
    lines.append("\n".join(term_strs) + ";")
    lines.append("}")
    return "\n".join(lines)


def main():
    if len(sys.argv) == 1:
        print(__doc__.strip())
        sys.exit(0)
    max_dim = 6
    out = None
    args = sys.argv[1:]
    while args:
        if args[0] == '--max' and len(args) > 1:
            max_dim = int(args[1]); args = args[2:]
        elif args[0] == '--out' and len(args) > 1:
            out = args[1]; args = args[2:]
        else:
            sys.exit(f"Unknown argument: {args[0]}\n{__doc__}")

    lines = ['#pragma once', '']
    for n in range(1, max_dim + 1):
        lines.append(gen_function(n))
        lines.append('')
    output = '\n'.join(lines)

    if out:
        with open(out, 'w') as f:
            f.write(output)
    else:
        print(output)

main()
