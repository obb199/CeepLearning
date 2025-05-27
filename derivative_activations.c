#include "derivative_activations.h"

bool derivative_relu(matrix * grads){ 
    if (grads == NULL) return false;

    for(int i = 0; i < grads->rows; i++){
        for(int j = 0; j < grads->cols; j++){
            if (grads->values[i][j] < 0) grads->values[i][j] = 0;
        }
    }
    return true;
}
