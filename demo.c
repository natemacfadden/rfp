#define RANDFAN_IMPLEMENTATION
#include "randfan.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// this is a demo of randfan.h

int main(int argc, char **argv) {
    (void)argc;

    // parse input vectors
    // -------------------
    // in row order. Ideally format "[[p0,p1,...],[q0,q1,..],...]"
    char* vecs_in = argv[1];
    int* vecs     = malloc(strlen(vecs_in) * sizeof(int)); // over-allocates

    int num_input =  0;
    int dim       = -1;

    int reading;
    while (*vecs_in != '\0') {
        // read the next number
        reading = isdigit(*vecs_in) || (*vecs_in == '+') || (*vecs_in == '-');
        if (reading) {
            vecs[num_input] = strtol(vecs_in, &vecs_in, 10);
            num_input++;
        }

        // check if we know the dimension
        if ((dim == -1) && ((*vecs_in == ')') || (*vecs_in == ']')))
            dim = num_input;

        // increment
        vecs_in++;
    }
    
    // check that lengths make sense, set #vectors if so
    if (num_input%dim != 0) {
        printf("Cannot infer dimension from input vectors...\n");
        printf("#ints=%d; #ints before first delimiter=%d\n",
            num_input, dim);
    }
    int num_vecs = num_input/dim;


    // get random fine, regular triangulation
    // --------------------------------------
    // (just call randfan)
    int max_num_simps = 100000;
    uint32_t* simps   = malloc(max_num_simps * sizeof(uint32_t));
    int num_simps;

    uint64_t seed = 1102;
    int retval = randfan(
        vecs, dim, num_vecs,
        max_num_simps, seed,
        simps, &num_simps);

    for (int i=0; i<num_simps; ++i) {
        printf("[");
        for (int j=0; j<dim; ++j) {
            printf("%d,", simps[dim* i+j]);
        }
        printf("],");
    }
    printf("\n");


    // free data
    free(vecs);
    free(simps);

    return 0;
}
