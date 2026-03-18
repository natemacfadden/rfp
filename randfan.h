#ifndef RANDFAN_H
#define RANDFAN_H

// HEADER
// ======
#include <stdint.h>

/*
**Description:**
Constructs a random regular fan. I.e., a random regular triangulation of the
vector configuration defined by the input vectors.

Does so via pushing-style arguments.

**Arguments:**
- `vecs`:          The input vectors.
- `dim`:           The dimension of the vector configuration.
- `num_vecs`:      The number of vectors input.
- `max_num_simps`: Max allowed number of simplices - to prevent writing out of
                   simps container.
// rng
- `seed`:          A seed for xoshiro128** RNG. First passed through splitmix64.
// output objects
- `simps`:         A container for the simplices.
- `num_simps`:     The number of simplices written.

**Returns:**
A status code according to following list:
    FILL IN
*/
int randfan(
    int * vecs,
    int dim,
    int num_vecs,
    int max_num_simps,
    uint64_t seed,
    uint32_t * simps,
    uint32_t * num_simps
);


// IMPLEMENTATION
// ==============
#ifdef RANDFAN_IMPLEMENTATION

#include <stdio.h>

// HELPER METHODS (copied in full; modifications marked)
// ==============
/*  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE. */

#include <stdint.h>

/* This is xoroshiro128** 1.0, one of our all-purpose, rock-solid,
   small-state generators. It is extremely (sub-ns) fast and it passes all
   tests we are aware of, but its state space is large enough only for
   mild parallelism.

   For generating just floating-point numbers, xoroshiro128+ is even
   faster (but it has a very mild bias, see notes in the comments).

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill s. */


static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}


//static uint64_t s[2]; -- removed for parallelism concerns in randfan
// (all methods below will have `void` argument changed to uint64_t s[2])

uint64_t next(uint64_t s[2]) {
    const uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    const uint64_t result = rotl(s0 * 5, 7) * 9;

    s1 ^= s0;
    s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); // a, b
    s[1] = rotl(s1, 37); // c

    return result;
}


/* This is the jump function for the generator. It is equivalent
   to 2^64 calls to next(); it can be used to generate 2^64
   non-overlapping subsequences for parallel computations. */

void jump(uint64_t s[2]) {
    static const uint64_t JUMP[] = { 0xdf900294d8f554a5, 0x170865df4b3201fc };

    uint64_t s0 = 0;
    uint64_t s1 = 0;
    for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
        for(int b = 0; b < 64; b++) {
            if (JUMP[i] & UINT64_C(1) << b) {
                s0 ^= s[0];
                s1 ^= s[1];
            }
            next(s);
        }

    s[0] = s0;
    s[1] = s1;
}


/* This is the long-jump function for the generator. It is equivalent to
   2^96 calls to next(); it can be used to generate 2^32 starting points,
   from each of which jump() will generate 2^32 non-overlapping
   subsequences for parallel distributed computations. */

void long_jump(uint64_t s[2]) {
    static const uint64_t LONG_JUMP[] = { 0xd2a98b26625eee7b, 0xdddf9b1090aa7ac1 };

    uint64_t s0 = 0;
    uint64_t s1 = 0;
    for(int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++)
        for(int b = 0; b < 64; b++) {
            if (LONG_JUMP[i] & UINT64_C(1) << b) {
                s0 ^= s[0];
                s1 ^= s[1];
            }
            next(s);
        }

    s[0] = s0;
    s[1] = s1;
}
/* End of xoshiro128** by Blackman and Vigna */

// for seeding xoshiro128**
uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

// Fisher-Yates
void fisher_yates(uint32_t * lis, uint32_t len, uint64_t* rng_state) {
    uint64_t j;

    for (int i=len-1; i>=0; --i) {
        j = next(rng_state) % (i+1); // get a random index to swap
        uint32_t tmp = lis[i]; lis[i] = lis[j]; lis[j] = tmp;
    }
}

// H-REP OF N-1 SIMPLEX
// ====================
int det(int * M, int n) {
    // base case
    if (n == 2)
        return M[2* 0+0]*M[2* 1+1] - M[2* 1+0]*M[2* 0+1];

    // recurse
    int out = 0;
    int M_trim[(n-1)*(n-1)];
    int sign = -1;
    for (int i=0; i<n; ++i) {
        // update the sign
        sign *= -1;

        // trim ith row and 0th column
        for (int itrim=0; itrim<n-1; ++itrim) {
            int jumped = (itrim >= i); // whether we skipped row-i
            for (int jtrim=0; jtrim<n-1; ++jtrim) {
                M_trim[(n-1)* itrim+jtrim] = M[n* (itrim+jumped)+(jtrim+1)];
            }
        }

        // Laplace expansion
        out += sign*M[n* i+0]*det(M_trim, n-1);
    }

    return out;
}

void hrep(int *R, int dim, int *x) {
    // computes a normal x to R... Rn=0
    // does so by setting x=(-1)^k det(R[:,:!=i])
    int R_trim[(dim-1)*(dim-1)];
    int sign = -1;
    for (int xi=0; xi<dim; ++xi) {
        // update the sign
        sign *= -1;

        // get the trimmed arr
        for (int jtrim=0; jtrim<dim-1; ++jtrim) {
            int jumped = (jtrim >= xi); // whether we skipped row-i
            for (int itrim=0; itrim<dim-1; ++itrim) {
                R_trim[(dim-1)* itrim+jtrim] = R[dim* itrim+(jtrim+jumped)];
            }
        }

        // set x[i]
        x[xi] = sign*det(R_trim,dim-1);
    }
}


// RANDFAN BEGINS
// ==============
int randfan(
    int * vecs,
    int dim,
    int num_vecs,
    int max_num_simps,
    uint64_t seed,
    uint32_t * simps,
    uint32_t * num_simps)
{
    /*
    **Description:**
    Constructs a random regular fan. I.e., a random regular triangulation of the
    vector configuration defined by the input vectors.

    Does so via pushing-style arguments.

    **Arguments:**
    - `vecs`:          The input vectors.
    - `dim`:           The dimension of the vector configuration.
    - `num_vecs`:      The number of vectors input.
    - `max_num_simps`: Max allowed number of simplices - to prevent writing out
                       of simps container.
    // rng
    - `seed`:          A seed for xoshiro128** RNG. First passed through
                       splitmix64.
    // output objects
    - `simps`:         A container for the simplices.
    - `num_simps`:     The number of simplices written.

    **Returns:**
    A status code according to following list:
        FILL IN
    */

    // seed the RNG
    uint64_t s[2];
    s[0] = splitmix64(&seed);
    s[1] = splitmix64(&seed);

    // indices
    uint32_t inds[num_vecs];
    for (uint32_t i = 0; i < num_vecs; i++)
        inds[i] = i;

    // Fisher-Yates shuffling
    fisher_yates(inds, num_vecs, s);

    // DEBUG PRINT NORMAL
    int normal[dim];
    hrep(vecs, dim, normal);
    for (int i=0; i<dim; ++i) {
        printf("%d,",normal[i]);
    }
    printf("\n");

    // DEBUG PRINT vecs
    int vi;
    printf("[");
    for (int ii=0; ii<num_vecs; ++ii) {
        vi = inds[ii]; // this allows the index list to shuffle

        // DEBUG
        if (1) {
            printf("[");
            for (int di=0; di<dim; ++di) {
                printf("%d,",vecs[di + vi*dim]);
            }
            printf("],");
        } else if (1) {
            printf("%d,", vi);
        }
        
    }
    printf("]\n");

    return 0;
}

#endif

#endif
