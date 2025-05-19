#ifndef _MATRIX_H
#define _MATRIX_H

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define M_E 2.7182818284590452354
#define MATRIX_TYPE double
#define PRINT_PRECISION "%.3f "

typedef struct{
    int rows;
    int cols;
    MATRIX_TYPE **values;
}matrix;

double** pointer_allocation(int, int); //check
bool pointer_desallocation(int, MATRIX_TYPE**); //check
bool matrix_init(int, int, matrix *); //check
bool matrix_desallocation(matrix *); //check
bool matrix_print(matrix *); //check
bool matrix_random_init(MATRIX_TYPE, MATRIX_TYPE, int, int, int, int, matrix *); //check
bool matrix_sum(matrix *, matrix *, matrix *); //check
bool matrix_subtraction(matrix *, matrix *, matrix *); //check
bool matrix_sum_columns(matrix *, matrix *); //check
bool matrix_sum_column_by_line(matrix *, matrix *, matrix *); //check
bool matrix_multiplication(matrix *, matrix *, matrix *); //check
bool matrix_multiplication_by_constant(matrix *, MATRIX_TYPE); //check
bool matrix_hadamart_product(matrix *, matrix *, matrix *); //check
bool matrix_transposition(matrix *, matrix *); //check
bool matrix_zeros_init(int, int, matrix *); //check
bool matrix_ones_init(int, int, matrix *); //check
bool matrix_identity_init(int, matrix *); //check
bool matrix_randomize_lines(int, int, matrix *); //check
bool matrix_minmax(matrix *); //check
bool matrix_copy(matrix *, matrix *); //check
bool matrix_reshape(matrix *, matrix *); //check

#endif // __MATRIX_H

