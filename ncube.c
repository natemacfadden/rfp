#include <assert.h>
#include <stdio.h>
#include <stdlib.h> 

// prints the n-dim unit cube.

int main(int argc, char **argv) {
    int n = atoi(argv[1]);
    assert(n <= 64);

    uint64_t upper = (n == 64) ? UINT64_MAX : ((uint64_t)1 << n) - 1;

    for (uint64_t i = 0; i<=upper; ++i) {
        printf("[");
        for (int j=n-1; j>=0; --j)
            printf("%llu,",1&(i>>j));
        printf("1], "); // end with 1 (to homogenize)
    }
    printf("\n");

    return 0;
}
