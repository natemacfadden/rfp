#ifndef RANDFAN_H
#define RANDFAN_H

//#define DEBUG

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
- `seed`:          A seed for xoshiro256++ RNG. First passed through splitmix64.
// output objects
- `simps`:         A container for the simplices.
- `num_simps`:     The number of simplices written.

**Returns:**
A status code according to following list:
    0:  success
    -1: memory allocation problem
    -2: couldn't find seed simplex
    FILL IN
*/
int randfan(
    int *vecs,
    int dim,
    int num_vecs,
    int max_num_simps,
    uint64_t seed,
    uint32_t *simps,
    int *num_simps
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
    for(int i = 0; i < (int)(sizeof JUMP / sizeof *JUMP); i++)
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
    for(int i = 0; i < (int)(sizeof LONG_JUMP / sizeof *LONG_JUMP); i++)
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

// for seeding xoshiro256++
uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

// Fisher-Yates (shuffle list)
void fisher_yates(int * lis, int len, uint64_t* rng_state) {
    uint64_t j;

    for (int i=len-1; i>=0; --i) {
        j = next(rng_state) % (i+1); // get a random index to swap
        int tmp = lis[i]; lis[i] = lis[j]; lis[j] = tmp;
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

// REGFAN CODE
// ===========
// HELPER DATA STRUCTURES
// ----------------------
typedef struct {
    int labels[MAX_DIM];              // vertices of the simplex
    int normals[MAX_DIM*MAX_DIM];     // inwards-facing normals
    int external_facet_inds[MAX_DIM]; // indices of external facets
    int num_external_facets;
} Simplex;

// H-REP OF N-1 SIMPLEX
// --------------------
int det(int *M, int dim) {
    // computes the determinant of M, a dim-by-dim matrix

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

int dot(int *a, int *b, int dim) {
    int out = 0;
    for (int i=0; i<dim; ++i)
        out += a[i]*b[i];
    return out;
}

void hrep(int *R, int dim, int *H) {
    // get the inwards-facing H-representation of the simplex
    // (iterate over all facets, get the normal, orient so it faces inwards)
    // facets come in order R\i for i=0,1,2,...

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
        int dotted = dot(&H[dim*i],&R[dim*i],dim);

        if (dotted<0) {
            for (int k=0; k<dim; ++k)
                H[i*dim+k] *= -1;
        }

    }
}

int simp_contains(Simplex *simp, int *vecs, int dim, int *labels, int num_labels) {
    // check if the simplex `simp` contains any of the vectors in `labels`
    for (int ilabel=0; ilabel<num_labels; ++ilabel){
        int label = labels[ilabel];

        // nice check: skip if label is explicitly in simp
        int skip = 0;
        for (int isimp=0; isimp<dim; ++isimp) {
            if (label == simp->labels[isimp]) {skip = 1; break;}
        }
        if (skip == 1) continue;

        // check if label is in conical hull
        int bad = 1;
        int dotted;
        for (int ifacet=0; ifacet<dim; ++ifacet) {
            dotted = dot(
                &(simp->normals[MAX_DIM*ifacet]),
                &vecs[dim*label],
                dim);

            bad = bad && (dotted>=0);
        }

        // label *is* in conical hull :(
        if (bad) {
            return 1;
        }
    }

    // no label was in conical hull :)
    return 0;
}

// RANDFAN BEGINS
// ==============
int randfan(
    int *vecs,
    int dim,
    int num_vecs,
    int max_num_simps,
    uint64_t seed,
    uint32_t *simps,
    int *num_simps)
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
    - `seed`:          A seed for xoshiro256++ RNG. First passed through
                       splitmix64.
    // output objects
    - `simps`:         A container for the simplices.
    - `num_simps`:     The number of simplices written.

    **Returns:**
    A status code according to following list:
        0:  success
        -1: memory allocation problem
        -2: couldn't find initial simplex
        FILL IN
    */
    // set up some variables
    // ---------------------
    // misc
    int return_code = 0;
    uint64_t s[4]; // RNG state

    // vc variables
    int labels[num_vecs]; // labels, defined as 0,1,...,num_vecs-1
    for (int i = 0; i < num_vecs; i++) { labels[i] = i; }

    // simplex variables
    int _inds[dim]; // indices into the shuffled labels... defines simplex
    int simp_labels[dim];
    int seed_simp_R[dim*dim];
    int seed_simp_H[dim*dim];

    // fan variables
    Simplex *_simps      = NULL;
    int *external_isimp  = NULL;
    int *external_ifacet = NULL; 

    // initialize the fan variables
    *num_simps      = 0;
    _simps = malloc(max_num_simps * sizeof(Simplex)); // internal use
    if (_simps == NULL) { return_code = -1; goto end; }

    // seed the RNG
    // ------------
    s[0] = splitmix64(&seed);
    s[1] = splitmix64(&seed);
    s[2] = splitmix64(&seed);
    s[3] = splitmix64(&seed);

    // get an initial simplex
    // ----------------------
    // shuffle the labels using Fisher-Yates
    fisher_yates(labels, num_vecs, s);

    // begin with seed_simp = labels[0],labels[1],labels[2],...
    for (int i=0; i<dim; ++i) _inds[i] = i;

    // DEBUG PRINT
    #ifdef DEBUG
        printf("Looking for initial simplex...");
    #endif

    while (1) {
        // for retrying next iteration w/ goto
        begin_seed:;

        // get simplex labels, sort them
        for (int i = 0; i < dim; i++) simp_labels[i] = labels[_inds[i]];
        insertion_sort(simp_labels, dim);

        // DEBUG PRINT
        #ifdef DEBUG
            printf("[");
            for (int i=0; i<dim; ++i)
                printf("%d,",simp_labels[i]);
            printf("], ");
        #endif

        // get the H-representation
        for (int i=0; i<dim; ++i) {
            for (int j=0; j<dim; ++j) {
                seed_simp_R[dim* i+j] = vecs[dim* simp_labels[i]+j];
            }
        }
        hrep(seed_simp_R, dim, seed_simp_H);

        // write as formal simplex
        // -----------------------
        Simplex *simp = &_simps[0];

        for (int i=0; i<dim; ++i) {
            simp->labels[i] = simp_labels[i];

            simp->external_facet_inds[i] = i; // all facets begin as external
            for (int j=0; j<dim; ++j)
                simp->normals[MAX_DIM* i+j] = seed_simp_H[dim* i+j];
        }
        simp->num_external_facets = dim;

        // check if any other vector is included in this cone
        // --------------------------------------------------
        int cont = simp_contains(simp, vecs, dim, labels, num_vecs);
        
        if (cont) {
            // increment the indices corresponding to the simplex
            // iterate over _inds right-to-left, trying to increment each val
            int i = dim - 1;
            while (i >= 0 && _inds[i] >= num_vecs - dim + i) {
                // (second condition sees if we can update _inds[j] for j>i)
                i--;
            }

            // exhausted all combinations... error
            // (shouldn't ever hit though)
            if (i < 0) { return_code=-2; goto end; }

            // update the index i
            _inds[i]++;

            // update the indices j>i
            for (int j = i + 1; j < dim; j++)
                _inds[j] = _inds[i] + (j - i);
            goto begin_seed;
        }

        break;
    }

    // discard used labels
    int num_labels = num_vecs;
    for (int i=0; i<num_labels; ++i) {
        // check if we need to throw away the ith label
        for (int j=0; j<dim; ++j) {
            if (labels[i] == _simps[0].labels[j]) {
                // matches a label in simp... throw it away
                labels[i] = labels[num_labels-1];
                num_labels--;
                i--; // decrement i in case swapped-in matches another label
                break;
            }
        }
    }

    // update simp count
    (*num_simps)++;

    // build other simplices
    // ---------------------
    int external_numfacets;
    external_isimp  = malloc(max_num_simps * dim * sizeof(int));
    external_ifacet = malloc(max_num_simps * dim * sizeof(int));
    if (external_isimp == NULL) { return_code = -1; goto end; }
    if (external_ifacet == NULL) { return_code = -1; goto end; }

    while (num_labels > 0) {
        // re-shuffle the labels
        fisher_yates(labels, num_labels, s);

        // try pushing each label
        for (int ilabel=0; ilabel<num_labels; ++ilabel) {
            external_numfacets = 0;
            int label = labels[ilabel];

            // get geometric vector
            int v[dim];
            for (int i=0; i<dim; ++i) v[i] = vecs[dim* label+i];

            // COMPUTE VISIBLE FACETS
            for (int isimp=0; isimp<*num_simps; ++isimp) {
                Simplex *simp = &_simps[isimp];

                for (int ifacet=0; ifacet<simp->num_external_facets; ++ifacet) {
                    int dotted = dot(
                        &simp->normals[MAX_DIM*simp->external_facet_inds[ifacet]],
                        v,
                        dim);

                    if (dotted < 0) {
                        external_isimp[external_numfacets]  = isimp;
                        external_ifacet[external_numfacets] = ifacet;
                        external_numfacets++;
                    }
                }
            }

            return 0;

            // tentatively add visible facets
            // (don't update num_simps yet in case any of these are bad)
            for (int k=0; k<external_numfacets; ++k) {
                // grab at index >=numvecs
                *num_simps+k;
                Simplex *simp = &_simps[*num_simps+k];

                // collect the labels
                simp->labels[0] = label;
                int skipped = 0;
                for (int i=0; i<dim; ++i) {
                    // ith facet corresponds to deleting ith point
                    skipped = skipped || (i==external_ifacet[k]);
                    if (skipped) continue;
                    simp->labels[i-skipped+1] = _simps[external_isimp[k]].labels[i];
                }

                // set the external facets
                for (int i=1; i<dim; ++i) {
                    simp->external_facet_inds[i-0] = i; // all facets begin as external
                    for (int j=0; j<dim; ++j)
                        simp->normals[MAX_DIM* i+j] = seed_simp_H[dim* i+j];
                    simp->num_external_facets = dim;
                }
            }

            // DEBUG
            printf("isimp ");
            for (int i=0; i<external_numfacets; ++i) {
                printf("%d,",external_isimp[i]);
            }
            printf("\n");
            printf("ifacet ");
            for (int i=0; i<external_numfacets; ++i) {
                printf("%d,",external_ifacet[i]);
            }
            printf("\n");
            /*
            for simp in simps:
                for i in range(num_external_facets):
                    n = simp.facets[i]
            */

            // CHECK IF SIMPLEX cup labels[ilabel] CONTAINS ANYONE ELSE

            // IF YES, TRY NEXT ilabel
            // IF NO, ADD THESE SIMPLICES
        }

        return 0;
    }

    // end goto
    // --------
    end:
        free(_simps);
        free(external_isimp);
        free(external_ifacet);
        return return_code;
}

#endif

#endif
