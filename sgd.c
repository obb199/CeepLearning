#include "sgd.h"

bool SGD_apply_grads(struct dense_layer * layer){
    matrix *d_weights = malloc(sizeof(matrix));
    matrix *d_biases = malloc(sizeof(matrix));
    matrix tranposed_inputs;
    matrix *new_weights = malloc(sizeof(matrix));
    matrix_init(layer->n_inputs, layer->n_outputs, new_weights);
    matrix *new_biases = malloc(sizeof(matrix));
    matrix_init(1, layer->n_outputs, new_biases);
    
    matrix_init(layer->n_inputs, layer->n_outputs, d_weights);
    matrix_init(1, layer->n_outputs, d_biases);
    matrix_init(layer->inputs->cols, layer->inputs->rows, &tranposed_inputs);
    matrix_transposition(layer->inputs, &tranposed_inputs);
    
    matrix_multiplication(&tranposed_inputs, layer->input_grads, d_weights);
    matrix_multiplication_by_constant(d_weights, layer->learning_rate);
    matrix_subtraction(layer->weights, d_weights, new_weights); //new weights computed
    matrix_desallocation(layer->weights); //desalloc of old weights
    layer->weights = new_weights; //setting new weights
    
    matrix_sum_columns(layer->input_grads, d_biases);
    matrix_multiplication_by_constant(d_biases, layer->learning_rate);
    matrix_subtraction(layer->biases, d_biases, new_biases); //new biases computed
    matrix_desallocation(layer->biases);
    layer->biases = new_biases;
    
}
