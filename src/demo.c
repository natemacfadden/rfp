#define RFP_IMPLEMENTATION
#include "rfp.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Demo of RFP - code for generating random, fine, pushing triangulations. */

void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [OPTIONS] [POINTS]\n"
        "\n"
        "Options:\n"
        "  -n, --num <int>          Number of triangulations (default: 1)\n"
        "  -s, --seed <uint64>      RNG seed (default: 1102)\n"
        "  --maxnumsimps <int>      Max simplices allocated (default: 100000)\n"
        "\n"
        "Points may be passed as a string argument or piped via stdin.\n",
        prog
    );
}

int main(int argc, char **argv) {
    // default variables
    int num_triangs   = 1;
    uint64_t rng_seed = 1102;
    int max_num_simps = 100000;

    // parse inputs
    // ------------
    // vectors in row order. Ideally format "[[p0,p1,...],[q0,q1,..],...]"
    char* vecs_in = NULL;
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--num") == 0) ||
            (strcmp(argv[i], "-n") == 0)) {
            // user set the number of triangulations
            num_triangs = (int)strtol(argv[++i], NULL, 10);

        } else if ((strcmp(argv[i], "--seed") == 0) ||
                   (strcmp(argv[i], "-s") == 0)) {
            // user set RNG seed
            rng_seed = strtoull(argv[++i], NULL, 10);

        } else if (strcmp(argv[i], "--maxnumsimps") == 0) {
            // user set max number of simplices
            max_num_simps = (int)strtol(argv[++i], NULL, 10);

        } else if (argv[i][0] == '-') {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            exit(1);
        } else {
            // user passed in points/vectors as a string
            vecs_in = argv[i];
        }
    }

    // (vectors not set from string... read from stdin)
    if (!vecs_in) {
        char buf[1 << 20];
        int len = fread(buf, 1, sizeof(buf) - 1, stdin);

        if (len == sizeof(buf) - 1) {
            fprintf(stderr, "Input too large, truncated\n");
            exit(1);
        }
        buf[len] = '\0';
        vecs_in = buf;
    }

    // convert the vectors to integers
    // -------------------------------
    int* vecs     = malloc(strlen(vecs_in) * sizeof(int)); // over-allocates

    int num_input =  0;
    int dim       = -1;

    while (*vecs_in != '\0') {
        // read the next number
        if (isdigit(*vecs_in) || (*vecs_in == '+') || (*vecs_in == '-')) {
            vecs[num_input] = strtol(vecs_in, &vecs_in, 10);
            num_input++;
        }

        // check if we know the dimension
        if ((dim == -1) && ((*vecs_in == ')') || (*vecs_in == ']')))
            dim = num_input;

        // increment
        vecs_in++;
    }

    if (dim <= 0) {
        fprintf(stderr, "Cannot infer dimension from input\n");
        usage(argv[0]);
        exit(1);
    }

    // check that lengths make sense, set #vectors if so
    if (num_input%dim != 0) {
        fprintf(stderr, "Cannot infer dimension from input vectors...\n");
        fprintf(stderr, "#ints=%d; #ints before first delimiter=%d\n",
            num_input, dim);
        usage(argv[0]);
        exit(1);
    }
    int num_vecs = num_input/dim;

    // get random fine, regular triangulation
    // --------------------------------------
    uint32_t* simps   = malloc(max_num_simps * sizeof(uint32_t));
    int num_simps;

    for (int itriang=0; itriang<num_triangs; ++itriang) {
        int retval = rfp(
            vecs, dim, num_vecs,
            max_num_simps, &rng_seed,
            simps, &num_simps);

        if (retval != 0) { printf("return code %d...\n", retval); continue; }

        for (int i=0; i<num_simps; ++i) {
            printf("[");
            for (int j=0; j<dim; ++j) {
                printf("%d,", simps[dim* i+j]);
            }
            printf("],");
        }
        printf("\n");
    }

    // free data and exit
    // ------------------
    free(vecs);
    free(simps);

    return 0;
}
