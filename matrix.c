#include "matrix.h"

double** pointer_allocation(int rows, int cols){
    if (rows <= 0 || cols <= 0){
        return NULL;
    }

    double **matrix_pointer = malloc(rows * sizeof(double *));

    if (matrix_pointer == NULL){
      return NULL;
    }
    
    for (int i = 0; i < rows; i++){
        matrix_pointer[i] = malloc(cols * sizeof(double));
        if (matrix_pointer[i] == NULL) return NULL;
    }

    return matrix_pointer;
}


bool pointer_deallocation(int rows, double **matrix_pointer) {
    if (rows <= 0 || matrix_pointer == NULL) return false;

    for (int i = 0; i < rows; i++) {
        if (matrix_pointer[i] != NULL) {
            free(matrix_pointer[i]);
            matrix_pointer[i] = NULL;
        }
    }
    free(matrix_pointer);
    return true;
}


bool matrix_init(int rows, int cols, matrix *m){
    if (rows <= 0 || cols <= 0 || m == NULL){
        return false;
    }

    m->rows = rows;
    m->cols = cols;
    m->values = pointer_allocation(rows, cols);

    if (m->values != NULL) return true;
    return false;
}


bool matrix_desallocation(matrix *m) {
    if (m == NULL || m->values == NULL) return false;

    pointer_deallocation(m->rows, m->values);
    m->values = NULL; // Set to NULL to avoid dangling pointer in the structure

    return true;
}


bool matrix_print(matrix * m){
    if (m == NULL){
        return false;
    }

    for (int i = 0; i < m->rows; i++){
        for (int j = 0; j < m->cols; j++){
            printf(PRINT_PRECISION, m->values[i][j]);
        }
        printf("\n");
    }
    
    printf("\n");

    return true;
}

bool matrix_random_init(double min_value, double max_value, int seed, int precision, int rows, int cols, matrix *m){
    if (max_value == min_value || m == NULL){
        return false;
    }

    matrix_init(rows, cols, m);

    if (min_value > max_value){
        int aux = min_value;
        max_value = min_value;
        min_value = aux;
    }

    if (seed != -1){
        srand(seed);
    }
    
    int pot = 10;
    for (int i = 0; i < precision; i++) pot *= 10;

    double fractional_part_of_min_value = min_value - (int)min_value;
    double fractional_part_of_max_value = max_value - (int)max_value;
    const double fractional_range = fractional_part_of_max_value - fractional_part_of_min_value;
    const int int_range = (int)max_value - (int)min_value;
    
    #pragma omp parallel for
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            m->values[i][j] = rand()%(int_range) + (int)min_value; //integer part
            m->values[i][j] += (float)(rand()%(int)(pot))/pot + (fractional_part_of_min_value - fractional_part_of_max_value); //fractional part
        }
    }

    return true;
}


bool matrix_sum(matrix *m1, matrix *m2, matrix *m3){
    if (m1 == NULL || m2 == NULL || m3 == NULL ||
        m1->rows != m2->rows || m1->cols != m2->cols ||
        m1->rows != m3->rows || m1->cols != m3->cols){
        return false;
    }
    
    const int rows = m1->rows;
    const int cols = m1->cols;
    
    #pragma omp parallel for
    for(int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            m3->values[i][j] = m1->values[i][j] + m2->values[i][j];
        }
    }

    return true;
}


bool matrix_subtraction(matrix *m1, matrix *m2, matrix *m3){
    if (m1 == NULL || m2 == NULL || m3 == NULL ||
        m1->rows != m2->rows || m1->cols != m2->cols ||
        m1->rows != m3->rows || m1->cols != m3->cols){
        return false;
    }
    
    const int rows = m1->rows;
    const int cols = m1->cols;
    
    #pragma omp parallel for
    for(int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            m3->values[i][j] = m1->values[i][j] - m2->values[i][j];
        }
    }

    return true;
}


bool matrix_sum_columns(matrix *m1, matrix *m2){
    if(m2->rows != 1|| m1->cols != m2->cols ||
       m1 == NULL || m2 == NULL){
        return false;
    }

    if (!matrix_zeros_init(1, m1->cols, m2)) return false;
    
    const int rows = m1->rows;
    const int cols = m1->cols;
    
    #pragma omp parallel for
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            m2->values[0][j] += m1->values[i][j];
        }
    }

    return true;
}


bool matrix_sum_column_by_line(matrix *m1, matrix *m2, matrix *m3){
    if(m1->rows != m3->rows ||
       m1->cols != m3->cols ||
       m1->cols != m2->cols ||
       m2->rows != 1){
        return false;
    }
    
    const int rows = m1->rows;
    const int cols = m1->cols;
    
    #pragma omp parallel for
    for (int i = 0; i < rows ; i++){
        for (int j = 0; j < cols; j++){
            m3->values[i][j] = m1->values[i][j] + m2->values[0][j];
        }
    }

    return true;
}


bool matrix_multiplication(matrix *m1, matrix *m2, matrix *m3){
    if (m1 == NULL || m2 == NULL || m3 == NULL ||
        m1->cols != m2->rows ||
        m1->rows != m3->rows ||
        m2->cols != m3->cols){
        return false;
    }
    
    const int i_limit = m1->rows;
    const int j_limit = m2->cols;
    const int k_limit = m1->cols;
    
    #pragma omp parallel for
    for (register int i = 0; i < i_limit; i++){
        for (register int j = 0; j < j_limit; j++){
            for (register int k = 0; k < k_limit; k++){
                m3->values[i][j] +=  m1->values[i][k] * m2->values[k][j];
            }
        }
    }

    return true;
}


bool matrix_multiplication_by_constant(matrix *m1, double constant){
    if (m1 == NULL){
        return false;
    }
  
    const int rows = m1->rows;
    const int cols = m1->cols;
    
    #pragma omp parallel for
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            m1->values[i][j] = constant*m1->values[i][j];
        }
    }

    return true;
}


bool matrix_hadamart_product(matrix *m1, matrix *m2, matrix *m3){
    if (m1 == NULL || m2 == NULL || m3 == NULL ||
        m1->rows != m2->rows || m1->rows != m3->rows ||
        m1->cols != m2->cols || m1->cols != m3->cols){
        return false;
    }
    
    const int rows = m1->rows;
    const int cols = m1->cols;
    
    #pragma omp parallel for
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            m3->values[i][j] = m1->values[i][j] * m2->values[i][j];
        }
    }

    return true;
}


bool matrix_transposition(matrix *m1, matrix *m2){
    if (m1 == NULL || m2 == NULL ||
        m1->rows != m2->cols || m1->cols != m2->rows){
        return false;
    }
    
    const int rows = m1->rows;
    const int cols = m1->cols;
    
    #pragma omp parallel for
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            m2->values[j][i] = m1->values[i][j];
        }
    }

    return true;
}


bool matrix_zeros_init(int rows, int cols, matrix *m1){
    if (m1 == NULL){
        return false;
    }

    matrix_init(rows, cols, m1);
    
    #pragma omp parallel for
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            m1->values[i][j] = 0;
        }
    }

    return true;
}



bool matrix_ones_init(int rows, int cols, matrix *m1){
    if (m1 == NULL){
        return false;
    }

    matrix_init(rows, cols, m1);
    
    #pragma omp parallel for
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            m1->values[i][j] = 1.0;
        }
    }

    return true;
}


bool matrix_identity_init(int dim, matrix *m1){
    if (m1 == NULL || dim < 1){
        return false;
    }

    matrix_init(dim, dim, m1);
    
    #pragma omp parallel for
    for (int i = 0; i < dim; i++){
        for (int j = 0; j < dim; j++){
            if (i == j){
                m1->values[i][j] = 1;
            }else{
                m1->values[i][j] = 0;
            }
        }
    }

    return true;
}


bool matrix_randomize_lines(int iterations, int seed, matrix *m1){
    if (m1 == NULL || iterations <= 0){
        return false;
    }

    if (seed != -1){
        srand(seed);
    }else{
        srand(time(NULL));
    }

    int line1 = 0, line2 = 0;
    double aux;
    
    for(int i = 0; i < iterations; i++){
        line1 = rand()%m1->rows;
        line2 = rand()%m1->rows;

        for (int j = 0; j < m1->cols; j++){
            aux = m1->values[line1][j];
            m1->values[line1][j] = m1->values[line2][j];
            m1->values[line2][j] = aux;
        }
    }

    return true;
}


bool matrix_minmax(matrix *m1){
    if (m1 == NULL){
        return false;
    }

    double max_values_per_column[m1->cols];
    double min_values_per_column[m1->cols];

    for (int i = 0; i < m1->cols; i++){
        max_values_per_column[i] = m1->values[0][i];
        min_values_per_column[i] = m1->values[0][i];
    }

    for(int i = 0; i < m1->rows; i++){
        for (int j = 0; j < m1->cols; j++){
            if (max_values_per_column[j] < m1->values[i][j]) max_values_per_column[j] = m1->values[i][j];
            if (min_values_per_column[j] > m1->values[i][j]) min_values_per_column[j] = m1->values[i][j];
        }
    }

    for (int i = 0; i < m1->rows; i++){
        for (int j = 0; j < m1->cols; j++){
            m1->values[i][j] = (m1->values[i][j] - min_values_per_column[j])/(max_values_per_column[j] - min_values_per_column[j]);
        }
    }

    return true;
}


bool matrix_copy(matrix *m1, matrix *m2){
    if (m1 == NULL || m2 == NULL ||
        m1->rows != m2->rows ||
        m1->cols != m2->cols){
        return false;
    }

    for (int i = 0; i < m1->rows; i++){
        for (int j = 0; j < m1->cols; j++){
            m2->values[i][j] = m1->values[i][j];
        }
    }

    return true;
}


bool matrix_reshape(matrix *m1, matrix *m2){
    if (m1 == NULL || m2 == NULL ||
        m1->rows*m1->cols != m2->rows*m2->cols){
         return false;
    }

    int actual_line = 0;
    int actual_col = 0;

    for(int i = 0; i < m1->rows; i++){
        for(int j = 0; j < m1->cols; j++){
            m2->values[actual_line][actual_col] = m1->values[i][j];
            if (actual_col == m2->cols-1){
                actual_col = 0;
                actual_line += 1;
            }else{
                actual_col++;
            }
        }
    }

    return true;
}

bool matrix_copy_elements(matrix * m1, matrix m2){
    if (m1 == NULL || m1->rows != m2.rows || m1->cols != m2.cols) return false;
    
    const int rows = m1->rows;
    const int cols = m1->cols;
    
    #pragma omp parallel for
    for(int i = 0; i < rows; i++){
      for(int j = 0; j < cols; j++){
          m1->values[i][j] = m2.values[i][j];
        }
    }
    
    return true;
}
