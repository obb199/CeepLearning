#include "dense_layer.h"
#include "losses.h"
#include "derivative_losses.h"
#include "sgd.h"

int main(int argc, char * argv[]){
    srand(time(0));
    matrix input;
    matrix_init(32, 5, &input);
    matrix output;
    
    matrix_random_init(-5, 5, -1, 5, 32, 5, &input);
    matrix_init(32, 1, &output);
    
    for (int i = 0; i < 32; i++){
        output.values[i][0] = 0;
        for(int j = 0 ; j < 5; j++) output.values[i][0] += input.values[i][j];
    }
    
    matrix derivative_mse;
    matrix_init(32, 1, &derivative_mse);
    struct dense_layer l1;
    dense_layer_init(&l1, 5, 1, 0.0001);
    
    for (int i = 0; i < 50; i++){
      l1.forward(&l1, &input);
      printf("%f\n", mean_squared_error(l1.outputs, &output));
      derivative_mean_squared_error(l1.outputs, &output, &derivative_mse);
      l1.backward(&l1, &derivative_mse);
      SGD_apply_grads(&l1);
    }
    
    return 0;
}
