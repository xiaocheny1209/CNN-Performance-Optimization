#include "attention.h"
#include <stdio.h>
#include <math.h>

// Helper function to allocate a matrix of size rows x cols
float **allocate_matrix_1(int rows, int cols)
{
    float **matrix = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++)
    {
        matrix[i] = (float *)malloc(cols * sizeof(float));
    }
    return matrix;
}

// Helper function to free a matrix
void free_matrix_1(float **matrix, int rows)
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
    float **result = allocate_matrix_1(cols, rows);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

void print2DFloatArray(float **array, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%f ", (float)array[i][j]); // Print each float with a space
        }
        printf("\n"); // Move to the next line after printing a row
    }
}

// Scaled dot-product attention
float **scaled_dot_product_attention(float **Q, float **K, float **V, int seqLength, int depth)
{
    // Step 1: Compute QK^T (dot product of Q and the transpose of K)
    float **K_T = transpose(K, seqLength, depth); // K transpose
    float **QK_T = matmul_blocking(Q, K_T, seqLength, depth, depth, seqLength);

    // Step 2: Scale the dot product by sqrt(depth)
    float scale_factor = 1.0 / sqrt(depth);
    for (int i = 0; i < seqLength; i++)
    {
        for (int j = 0; j < seqLength; j++)
        {
            QK_T[i][j] = QK_T[i][j] * scale_factor;
        }
    }

    // Step 3: Apply softmax to the scaled dot product row-wise
    for (int i = 0; i < seqLength; i++)
    {
        float *softmax_output = softmax(QK_T[i], seqLength);
        free(QK_T[i]);            // Free the original row memory
        QK_T[i] = softmax_output; // Assign softmax result back to the matrix
    }

    // Step 4: Multiply by the value matrix V using blocking (tiled) matrix multiplication
    float **output = matmul_blocking(QK_T, V, seqLength, seqLength, seqLength, depth);
    free_matrix_1(K_T, depth);
    free_matrix_1(QK_T, seqLength);

    return output;
}
