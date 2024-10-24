#include "attention.h"
#include <stdio.h>
#include <math.h>

// Helper function to allocate a matrix of size rows x cols
float **allocate_matrix(int rows, int cols)
{
    float **matrix = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++)
    {
        matrix[i] = (float *)malloc(cols * sizeof(float));
    }
    return matrix;
}

// Helper function to free a matrix
void free_matrix(float **matrix, int rows)
{
    for (int i = 0; i < rows; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
}

// Transpose of a matrix
float **transpose(float **matrix, int rows, int cols)
{
    float **result = allocate_matrix(cols, rows);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

// Matrix multiplication: A (rows_A x cols_A) * B (cols_A x cols_B)
float **matrix_multiply(float **A, float **B, int rows_A, int cols_A, int cols_B)
{
    float **result = allocate_matrix(rows_A, cols_B);
    for (int i = 0; i < rows_A; i++)
    {
        for (int j = 0; j < cols_B; j++)
        {
            result[i][j] = 0;
            for (int k = 0; k < cols_A; k++)
            {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

// Apply softmax to a vector
void softmax(float *input, int length)
{
    float max = input[0];
    for (int i = 1; i < length; i++)
    {
        if (input[i] > max)
            max = input[i];
    }

    float sum = 0;
    for (int i = 0; i < length; i++)
    {
        input[i] = expf(input[i] - max);
        sum += input[i];
    }

    for (int i = 0; i < length; i++)
    {
        input[i] /= sum;
    }
}

// Apply softmax row-wise on a matrix
void softmax_matrix(float **matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        softmax(matrix[i], cols);
    }
}

// Scaled dot-product attention
float **scaled_dot_product_attention(float **Q, float **K, float **V, int seqLength, int depth)
{
    // Step 1: Compute QK^T (dot product of Q and the transpose of K)
    float **K_T = transpose(K, seqLength, depth);
    float **QK_T = matrix_multiply(Q, K_T, seqLength, depth, seqLength);

    // Step 2: Scale the dot product by sqrt(depth)
    float scale_factor = 1.0 / sqrtf(depth);
    for (int i = 0; i < seqLength; i++)
    {
        for (int j = 0; j < seqLength; j++)
        {
            QK_T[i][j] *= scale_factor;
        }
    }

    // Step 3: Apply softmax to the scaled dot product
    softmax_matrix(QK_T, seqLength, seqLength);

    // Step 4: Multiply by the value matrix V
    float **output = matrix_multiply(QK_T, V, seqLength, seqLength, depth);

    // Free the intermediate matrices
    free_matrix(K_T, depth);
    free_matrix(QK_T, seqLength);

    return output;
}
