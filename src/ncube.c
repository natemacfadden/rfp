#include <stdio.h>
#include <stdlib.h>

#include <inttypes.h>

/* prints the vertices of the n-dim unit cube [0,1]^n
   accepts a single (required) argument - the dimension n */

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Expects single argument\n");
        exit(1);
    }

    // parse the dimension
    int n = (int)strtol(argv[1], NULL, 10);
    if (n < 1) {
        fprintf(stderr, "Dimension n must be >=1\n");
        exit(1);
    } else if (n > 64) {
        fprintf(stderr, "Dimension n must be <=64\n");
        exit(1);
    } else if (n > 20) {
        fprintf(stderr, "You are requesting 2^%d vertices... reconsider\n", n);
        exit(1);
    }

    // vertices corresponds to bits
    uint64_t upper = (n == 64) ? UINT64_MAX : ((uint64_t)1 << n) - 1;
    for (uint64_t i = 0; i<=upper; ++i) {
        printf("[");
        for (int j=n-1; j>=0; --j)
            printf("%" PRIu64 ",", 1&(i>>j));
        printf("1], "); // end with 1 (to homogenize)
    }
    printf("\n");

    return 0;
}
