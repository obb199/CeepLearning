#include "losses.h"

float mean_squared_error(matrix * y_pred, matrix * y_true){
    if (y_pred == NULL) return false;
    if (y_true == NULL) return false;
    if (y_true->rows != y_pred->rows) return false;
    if (y_true->cols != y_pred->cols) return false;
    
    float mse = 0;
    float diff;
    
    for (int i = 0; i < y_pred->rows; i++){
      for(int j = 0; j < y_pred->cols; j++){
        diff = y_pred->values[i][j] - y_true->values[i][j];
        mse += diff * diff;
      }  
    }
    
    return mse/y_pred->rows;
}
