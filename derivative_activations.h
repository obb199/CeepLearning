#ifndef _DERIVATIVES_ACTIVATIONS_H
#define _DERIVATIVES_ACTIVATIONS_H

#include "matrix.h"

bool derivative_relu(matrix *);
//bool derivative_leaky_relu(matrix *);
bool derivative_sigmoid(matrix *);
//bool derivative_softmax(matrix *);

#endif
