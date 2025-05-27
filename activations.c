#include "activations.h"

bool activation_relu(matrix * output){
    if (output == NULL) return false;

    for(int i = 0; i < output->rows; i++){
        for(int j = 0; j < output->cols; j++){
            if (output->values[i][j] < 0) output->values[i][j] = 0;
        }
    }
    
    return true;
}
