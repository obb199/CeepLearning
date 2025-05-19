#include "dense_layer.h"

int main(int argc, char * argv[]){

    dense_layer layer;
    dense_layer_init(&layer, 4, 5, 0.1);
    matrix input;
    matrix_random_init(5, 10, 88, 8, 4, 5, &input);
    
    feedforward(&layer, input);

    matrix_print(&layer.input);
    printf("\n");
    matrix_print(&layer.output);

    return 0;
}
