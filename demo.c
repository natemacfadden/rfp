#define RANDFAN_IMPLEMENTATION
#include "randfan.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// this is a demo of regfan.h

int main(int argc, char **argv) {
    // parse input vectors
    // -------------------
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
    
    // check that lengths make sense, read #vectors if so
    if (num_input%dim != 0) {
        printf("Cannot infer dimension from input vectors...\n");
        printf("#ints=%d; #ints before first delimiter=%d\n",
            num_input, dim);
    }
    int num_vecs = num_input/dim;

    // debug
    for (int i=0; i<num_input; ++i){
        printf("%d,",vecs[i]);
        if (i%dim==dim-1)
            printf("\n");
    }

    return 0;
}
