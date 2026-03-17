#define RANDFAN_IMPLEMENTATION
#include "randfan.h"

#include <stdio.h>

// this is a demo of regfan.h

int main(int argc, char **argv) {
    // parse input vectors
    int num_vecs =  0;
    int dim      = -1;
    char* vecs_in = argv[1];
    int* vecs;

    while (*vecs_in != '\0') {
        printf("%c,", *vecs_in);
        vecs_in++;
    }
    printf("\n");

    return 0;
}
