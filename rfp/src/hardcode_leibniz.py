# Written by Claude Code (claude-sonnet-4-6)
# (w/ a little help from Nate)

"""Generate hardcoded C det functions for dimensions 1..N via Leibniz formula.

Usage: python3 hardcode_leibniz.py [--max <int>] [--out <file>]
  --max <int>   Maximum dimension (default: 6)
  --out <file>  Output file (default: det.h)
"""

import sys
from itertools import permutations

def gen_terms(n):
    """
    Return list of (sign, [col indices]) for Leibniz expansion of an nxn det.
    """
    terms = []
    for p in permutations(range(n)):
        # sign = (-1)^(# even-length cycles in p)
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
    # emit e.g. "+ M[0*3+1]*M[1*3+0]*M[2*3+2]"
    factors = [f"M[{row}*{n}+{col}]" for row, col in enumerate(cols)]
    product = "*".join(factors)
    if sign == 1:
        return f"+ {product}"
    else:
        return f"- {product}"


def gen_function(n):
    # build the C function as a list of lines
    terms = gen_terms(n)
    lines = []
    lines.append(f"#define DET{n}")
    lines.append(f"static int det{n}(int *M) {{")
    lines.append(f"    return")
    term_strs = [f"        {term_to_c(s, cols, n)}" for s, cols in terms]
    lines.append("\n".join(term_strs) + ";")
    lines.append("}")
    return "\n".join(lines)


def main():
    max_dim = 6
    out     = "det.h"

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

    with open(out, 'w') as f:
        f.write(output) 

main()
