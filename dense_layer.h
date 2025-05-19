#ifndef _DENSE_LAYER_H
#define _DENSE_LAYER_H

#include "matrix.h"

typedef struct{
    int n_inputs;
    int n_outputs;
    double learning_rate;
    matrix weights;
    matrix biases;
    matrix input;
    matrix output;
    
}dense_layer;

bool dense_layer_init(dense_layer *, int, int, double);
bool feedforward(dense_layer *, matrix);
bool backpropagation(dense_layer *, matrix *);




#endif
