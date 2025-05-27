#include "dense_layer.h"
#include "losses.h"
#include "derivative_losses.h"
#include "sgd.h"

int main(int argc, char * argv[]){
    srand(time(0));
    matrix input;
    matrix_init(32, 5, &input);
    matrix output;
    
    bool res1 = matrix_random_init(-5, 5, -1, 5, 32, 5, &input);
    bool res2 = matrix_init(32, 1, &output);
    
    for (int i = 0; i < 32; i++){
        output.values[i][0] = 0;
        for(int j = 0 ; j < 5; j++) output.values[i][0] += input.values[i][j];
    }

    dense_layer l1;
    bool res3 = dense_layer_init(&l1, 5, 1, 0.0001);
    bool res4 = feedforward(&l1, &input);

    feedforward(&l1, &input);
    
    
    matrix derivative_mse;
    matrix_init(32, 1, &derivative_mse);
    float mse = mean_squared_error(l1.outputs, &output);
    printf("%f\n", mse);
    bool res5 = derivative_mean_squared_error(l1.outputs, &output, &derivative_mse);
    bool res6 = backpropagation(&l1, &derivative_mse);
    
    SGD_apply_grads(&l1);
    
    //printf("\n%b %b %b %b %b %b", res1, res2, res3, res4, res5, res6);
    
    feedforward(&l1, &input);
    float mse1 = mean_squared_error(l1.outputs, &output);
    printf("%f\n", mse1);
    
    return 0;
}
