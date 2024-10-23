#include "matrix_ops.h"
#include <stdio.h>

float **matmul(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols)
{
    // validation check
    if (A_cols != B_rows)
        return NULL;

    // allocate memory for the result array
    float **result = (float **)malloc(A_rows * sizeof(float *));
    for (int i = 0; i < A_rows; i++)
    {
        result[i] = (float *)malloc(A_cols * sizeof(float));
    }

    for (int r = 0; r < A_rows; r++)
    {
        for (int c = 0; c < B_cols; c++)
        {
            result[r][c] = 0;
            for (int i = 0; i < A_cols; i++)
            {
                result[r][c] += A[r][i] * B[i][c];
            }
        }
    }

    return result;
}