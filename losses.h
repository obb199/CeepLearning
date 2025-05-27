#ifndef _LOSSES_H
#define _LOSSES_H
#include "matrix.h"

float mean_squared_error(matrix *, matrix *);
float mean_absolute_error(matrix *, matrix *);
float binary_categorical_crossentropy(matrix *, matrix *);
float categorical_crossentropy(matrix *, matrix *);

#endif
