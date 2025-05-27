#include "derivative_losses.h"

bool derivative_mean_squared_error(matrix * y_pred, matrix * y_true, matrix * grad_matrix){
    if (y_pred == NULL) return false;
    if (y_true == NULL) return false;
    if (grad_matrix == NULL) return false;
    if (y_true->rows != y_pred->rows) return false;
    if (y_true->cols != y_pred->cols) return false;
    if (grad_matrix->rows != y_true->rows) return false;
    if (grad_matrix->cols != y_true->cols) return false;
    
    for (int i = 0; i < y_pred->rows; i++){
      for(int j = 0; j < y_pred->cols; j++){
          grad_matrix->values[i][j] = 2 * (y_pred->values[i][j] - y_true->values[i][j]);
      }
    }
    
    return true;
}


