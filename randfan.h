#ifndef RANDFAN_H
#define RANDFAN_H



// copying could be reduced significantly in this code...



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

#ifndef MAX_DIM
#define MAX_DIM 8
#endif

#include <stdio.h>
#include <stdlib.h>

// EXTERNAL METHODS (copied in full; modifications marked)
// ----------------
/*  Written in 2019 by David Blackman and Sebastiano Vigna (vigna@acm.org)

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

/* This is xoshiro256++ 1.0, one of our all-purpose, rock-solid generators.
   It has excellent (sub-ns) speed, a state (256 bits) that is large
   enough for any parallel application, and it passes all tests we are
   aware of.

   For generating just floating-point numbers, xoshiro256+ is even faster.

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill s. */

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}


//static uint64_t s[4]; -- removed for parallelism concerns in randfan
// (all methods below will have `void` argument changed to uint64_t s[4])

uint64_t next(uint64_t s[4]) {
    const uint64_t result = rotl(s[0] + s[3], 23) + s[0];

    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;

    s[3] = rotl(s[3], 45);

    return result;
}


/* This is the jump function for the generator. It is equivalent
   to 2^128 calls to next(); it can be used to generate 2^128
   non-overlapping subsequences for parallel computations. */

void jump(uint64_t s[4]) {
    static const uint64_t JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };

    uint64_t s0 = 0;
    uint64_t s1 = 0;
    uint64_t s2 = 0;
    uint64_t s3 = 0;
    for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
        for(int b = 0; b < 64; b++) {
            if (JUMP[i] & UINT64_C(1) << b) {
                s0 ^= s[0];
                s1 ^= s[1];
                s2 ^= s[2];
                s3 ^= s[3];
            }
            next(s); 
        }
        
    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
    s[3] = s3;
}



/* This is the long-jump function for the generator. It is equivalent to
   2^192 calls to next(); it can be used to generate 2^64 starting points,
   from each of which jump() will generate 2^64 non-overlapping
   subsequences for parallel distributed computations. */

void long_jump(uint64_t s[4]) {
    static const uint64_t LONG_JUMP[] = { 0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635 };

    uint64_t s0 = 0;
    uint64_t s1 = 0;
    uint64_t s2 = 0;
    uint64_t s3 = 0;
    for(int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++)
        for(int b = 0; b < 64; b++) {
            if (LONG_JUMP[i] & UINT64_C(1) << b) {
                s0 ^= s[0];
                s1 ^= s[1];
                s2 ^= s[2];
                s3 ^= s[3];
            }
            next(s); 
        }
        
    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
    s[3] = s3;
}
/* End of xoshiro256++ by Blackman and Vigna */

// for seeding xoshiro128**
uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

// Fisher-Yates (shuffle list)
void fisher_yates(uint32_t * lis, uint32_t len, uint64_t* rng_state) {
    uint64_t j;

    for (int i=len-1; i>=0; --i) {
        j = next(rng_state) % (i+1); // get a random index to swap
        uint32_t tmp = lis[i]; lis[i] = lis[j]; lis[j] = tmp;
    }
}

// insertion sort
void insertion_sort(int *arr, int len) {
    for (int i = 1; i < len; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j+1] = arr[j];
            j--;
        }
        arr[j+1] = key;
    }
}

// REGFANS CODE
// ============
// HELPER DATA STRUCTURES
// ----------------------
typedef struct {
    int removed;         // index of removed vertex
    int normal[MAX_DIM]; // inward-facing normal
} Facet;

typedef struct {
    int labels[MAX_DIM];   // vertices of the simplex
    Facet facets[MAX_DIM]; // one facet per vertex
    int *external_facet_inds;
    int num_external_facets;
} Simplex;

// H-REP OF N-1 SIMPLEX
// --------------------
int det(int *M, int dim) {
    // computes the dimension of M, a dim-by-dim matrix

    // base case
    if (dim == 2)
        return M[2* 0+0]*M[2* 1+1] - M[2* 1+0]*M[2* 0+1];

    // recurse
    int out = 0;
    int M_trim[(dim-1)*(dim-1)];
    int sign = -1;
    for (int i=0; i<dim; ++i) {
        // update the sign
        sign *= -1;

        // trim ith row and 0th column
        for (int itrim=0; itrim<dim-1; ++itrim) {
            int jumped = (itrim >= i); // whether we skipped row-i
            for (int jtrim=0; jtrim<dim-1; ++jtrim) {
                M_trim[(dim-1)* itrim+jtrim] = M[dim* (itrim+jumped)+(jtrim+1)];
            }
        }

        // Laplace expansion
        out += sign*M[dim* i+0]*det(M_trim, dim-1);
    }

    return out;
}

void simp_facet_normal(int *R, int dim, int *x) {
    // for R the (row-wise) rays of a facet of a simplex, compute the normal x
    // i.e., the vector x such that Rx=0.
    // does so by setting x=(-1)^k det(R[:,:!=i])

    int R_trim[(dim-1)*(dim-1)];
    int sign = -1;
    for (int xi=0; xi<dim; ++xi) {
        // update the sign
        sign *= -1;

        // get the trimmed arr - skipping column-xi
        for (int jtrim=0; jtrim<dim-1; ++jtrim) {
            int jumped = (jtrim >= xi); // whether we skipped column-xi
            for (int itrim=0; itrim<dim-1; ++itrim) {
                R_trim[(dim-1)* itrim+jtrim] = R[dim* itrim+(jtrim+jumped)];
            }
        }

        // set x[i]
        x[xi] = sign*det(R_trim,dim-1);
    }
}

void hrep(int *R, int dim, int *H) {
    // get the inwards-facing H-representation of the simplex
    // (iterate over all facets, get the normal, orient so it faces inwards)

    int R_facet[(dim-1)*dim];
    for (int i=0; i<dim; ++i) {
        // get the rays corresponding to this facet
        for (int ifacet=0; ifacet<dim-1; ++ifacet) {
            int jumped = (ifacet >= i);
            for (int jfacet=0; jfacet<dim; ++jfacet) {
                R_facet[dim* ifacet+jfacet] = R[dim* (ifacet+jumped)+jfacet];
            }
        }

        // get the normal
        simp_facet_normal(R_facet, dim, &H[i*dim]);

        // ensure it is inwards-facing
        int dot  = 0;
        for (int k=0; k<dim; ++k)
            dot += H[i*dim + k] * R[dim* i+k];

        if (dot<0) {
            for (int k=0; k<dim; ++k)
                H[i*dim+k] *= -1;
        }

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
    // set up some variables
    // ---------------------
    int return_code = 0;

    uint32_t labels[num_vecs]; // labels, defined as 0,1,...,num_vecs-1
    for (uint32_t i = 0; i < num_vecs; i++) labels[i] = i;

    *num_simps      = 0;
    Simplex *_simps = malloc(max_num_simps * sizeof(Simplex)); // internal use

    // seed the RNG
    // ------------
    uint64_t s[4];
    s[0] = splitmix64(&seed);
    s[1] = splitmix64(&seed);
    s[2] = splitmix64(&seed);
    s[3] = splitmix64(&seed);

    // get an initial simplex
    // ----------------------
    int seed_simp_R[dim*dim];
    int seed_simp_H[dim*dim];

    // shuffle the labels using Fisher-Yates
    fisher_yates(labels, num_vecs, s);

    // begin with seed_simp = labels[0],labels[1],labels[2],...
    uint32_t _inds[dim]; // indices into the shuffled labels... defines simplex
    for (int i=0; i<dim; ++i) _inds[i] = i;

    printf("Looking for initial simplex...");
    while (1) {
        // for retrying next iteration w/ goto
        begin_loop:

        // get simplex labels
        int simp_labels[dim];
        for (int i = 0; i < dim; i++) simp_labels[i] = labels[_inds[i]];
        // sort them
        insertion_sort(simp_labels, dim);

        // check if the current simplex contains no other vectors
        printf("[");
        for (int i=0; i<dim; ++i)
            printf("%d,",simp_labels[i]);
        printf("], ");

        // get the H-representation
        for (int i=0; i<dim; ++i) {
            for (int j=0; j<dim; ++j) {
                seed_simp_R[dim* i+j] = vecs[dim* simp_labels[i]+j];
            }
        }
        hrep(seed_simp_R, dim, seed_simp_H);

        // check if any other vector is included in this cone
        for (int label=0; label<num_vecs; ++label){
            // skip if this label corresponds to one explicitly in the simp
            int skip = 0;
            for (int isimp=0; isimp<dim; ++isimp) {
                if (label == (int)simp_labels[isimp]) {
                    skip = 1;
                    break;
                }
            }
            if (skip == 1) continue;

            // label isn't explicitly in simp - check if it's in conical hull
            int bad = 1;
            for (int ifacet=0; ifacet<dim; ++ifacet) {
                int dot = 0;
                for (int k=0; k<dim; ++k)
                    dot += seed_simp_H[dim* ifacet+k] * vecs[dim* label+k];
                bad = bad && (dot>=0);
            }

            // label is geomerically inside :( update _inds and retry
            if (bad == 1){
                // increment the indices corresponding to the simplex
                for (int i=0; i<dim-1; ++i) {
                    if (_inds[i]+1 < _inds[i+1]) {
                        _inds[i] += 1;
                        goto begin_loop; // retry!
                    }
                }

                // no hits for i<dim-1... must set i=dim-1
                for (int i=0; i<dim-1; ++i) _inds[i] = i;
                _inds[dim-1] += 1;
                goto begin_loop; // retry!
            }
        }

        // no bad vectors!
        // save as formal simplex
        //_simps[*num_simps].labels              = 
        _simps[*num_simps].num_external_facets = dim;
        printf(":)\n");
        break;
    }

    end:
        free(_simps);
        return return_code;
}

#endif

#endif
