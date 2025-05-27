#include "dense_layer.h"

bool dense_layer_init(dense_layer * layer, int n_inputs, int n_outputs, double learning_rate){
    layer->n_inputs = n_inputs;
    layer->n_outputs = n_outputs;
    layer->learning_rate = learning_rate;
    
    matrix *weights = malloc(sizeof(matrix));
    matrix_init(n_inputs, n_outputs, weights);
    matrix_random_init(-1, 1, 42, 5, n_inputs, n_outputs, weights);
    layer->weights = weights;
    if (layer->weights == NULL) return false;
    
    matrix *biases = malloc(sizeof(matrix));
    matrix_init(1, n_outputs, biases);
    matrix_random_init(-1, 1, 422, 5, 1, n_outputs, biases);
    layer->biases = biases;
    if (layer->biases == NULL) return false;
    
    layer->inputs = NULL;
    layer->outputs = NULL;
    layer->input_grads = NULL;
    layer->output_grads = NULL;
    
    return true;  
}

bool feedforward(dense_layer * layer, matrix * inputs){
    if (layer == NULL) return false;
    if (inputs == NULL) return false;

    if (layer->inputs != NULL) matrix_desallocation(layer->inputs);
    if (layer->outputs != NULL) matrix_desallocation(layer->outputs);
    
    matrix *inputs_copy = malloc(sizeof(matrix));
    matrix_init(inputs->rows, inputs->cols, inputs_copy);
    for (int i = 0; i < inputs->rows; i++){
      for(int j = 0; j < inputs->cols; j++){
          inputs_copy->values[i][j] = inputs->values[i][j];
      }
    }
    
    layer->inputs = inputs_copy;
    
    matrix *output_matrix = malloc(sizeof(matrix));
    matrix_init(inputs->rows, layer->n_outputs, output_matrix);
    
    matrix_multiplication(inputs_copy, layer->weights, output_matrix); //weights multiplication
    
    for (int i = 0; i < inputs->rows; i++){
      for (int j = 0; j < layer->n_outputs; j++){
          output_matrix->values[i][j] += layer->biases->values[0][j]; //biases addition
        }
    }
    
    layer->outputs = output_matrix;
    
    if (layer->inputs == NULL) return false;
    if (layer->outputs == NULL) return false;
    
    return true;
}

bool backpropagation(dense_layer * layer, matrix * grads){
    if (layer == NULL) return false;
    if (grads == NULL) return false;
    
    if (layer->input_grads != NULL) matrix_desallocation(layer->input_grads);
    layer->input_grads = grads;
    if (layer->output_grads != NULL) matrix_desallocation(layer->output_grads);
    
    matrix *output_grads = malloc(sizeof(matrix));;
    matrix_init(grads->rows, layer->weights->rows, output_grads);
    
    matrix *transposed_weights = malloc(sizeof(matrix));
    matrix_init(layer->weights->cols, layer->weights->rows, transposed_weights);
    
    matrix_transposition(layer->weights, transposed_weights);
    
    matrix_multiplication(grads, transposed_weights, output_grads);
    
    layer->output_grads = output_grads;
    
    if (layer->input_grads == NULL) return false;
    if (layer->output_grads == NULL) return false;
    
    return true;
    
    
}
