#ifndef _DENSE_LAYER_H
#define _DENSE_LAYER_H

#include "matrix.h"

struct dense_layer{
    int n_inputs;
    int n_outputs;
    double learning_rate;
    matrix *weights;
    matrix *biases;
    matrix *inputs;
    matrix *outputs;
    matrix *input_grads;
    matrix *output_grads;
    
    bool (*forward)(struct dense_layer *, matrix *); 
    bool (*backward)(struct dense_layer *, matrix *);
    
};

bool dense_layer_init(struct dense_layer *, int, int, double);
bool dense_layer_feedforward(struct dense_layer *, matrix *);
bool dense_layer_backpropagation(struct dense_layer *, matrix *);


#endif
