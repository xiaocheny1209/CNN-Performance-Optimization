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

const int TILE_SIZE = 16;

// Matmul with blocking (loop tiling) optimization
float **matmul_blocking(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols)
{
    if (A_cols != B_rows)
        return NULL;

    float **result = (float **)malloc(A_rows * sizeof(float *));
    for (int i = 0; i < A_rows; i++)
    {
        result[i] = (float *)malloc(B_cols * sizeof(float));
        for (int j = 0; j < B_cols; j++)
        {
            result[i][j] = 0;
        }
    }

    // Tiled matrix multiplication
    for (int r_block = 0; r_block < A_rows; r_block += TILE_SIZE)
    {
        for (int c_block = 0; c_block < B_cols; c_block += TILE_SIZE)
        {
            for (int i_block = 0; i_block < A_cols; i_block += TILE_SIZE)
            {

                // Multiply the tiles
                for (int r = r_block; r < r_block + TILE_SIZE && r < A_rows; r++)
                {
                    for (int c = c_block; c < c_block + TILE_SIZE && c < B_cols; c++)
                    {
                        for (int i = i_block; i < i_block + TILE_SIZE && i < A_cols; i++)
                        {
                            result[r][c] += A[r][i] * B[i][c];
                        }
                    }
                }
            }
        }
    }

    return result;
}

// Structure to store CSR format
typedef struct
{
    float *values;
    int *col_indices;
    int *row_start;
    int nnz; // number of non-zero elements
} CSRMatrix;

// Convert dense matrix to CSR format
CSRMatrix *dense_to_csr(float **matrix, int rows, int cols)
{
    CSRMatrix *csr = (CSRMatrix *)malloc(sizeof(CSRMatrix));

    // Count non-zero elements
    int nnz = 0;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (matrix[i][j] != 0)
                nnz++;
        }
    }

    // Allocate CSR arrays
    csr->values = (float *)malloc(nnz * sizeof(float));
    csr->col_indices = (int *)malloc(nnz * sizeof(int));
    csr->row_start = (int *)malloc((rows + 1) * sizeof(int));
    csr->nnz = nnz;

    // Fill CSR arrays
    int index = 0;
    for (int i = 0; i < rows; i++)
    {
        csr->row_start[i] = index;
        for (int j = 0; j < cols; j++)
        {
            if (matrix[i][j] != 0)
            {
                csr->values[index] = matrix[i][j];
                csr->col_indices[index] = j;
                index++;
            }
        }
    }
    csr->row_start[rows] = nnz;

    return csr;
}

// Implement sparse matrix multiplication
float **matmul_sparse(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols)
{
    float **result;
    if (A_cols != B_rows)
        return NULL;

    /**** 1. Create CSR (compressed sparse row) format of input matrix ****/
    CSRMatrix *csr_A = dense_to_csr(A, A_rows, A_cols);

    result = (float **)malloc(A_rows * sizeof(float *));
    for (int i = 0; i < A_rows; i++)
    {
        result[i] = (float *)calloc(B_cols, sizeof(float));
    }

    /**** 2. Perform matrix multiplication on CSR format of input matrix ****/
    // When profiling, only loop over part 2.
    for (int i = 0; i < A_rows; i++)
    {
        for (int idx = csr_A->row_start[i]; idx < csr_A->row_start[i + 1]; idx++)
        {
            int colA = csr_A->col_indices[idx];
            float valA = csr_A->values[idx];

            for (int j = 0; j < B_cols; j++)
            {
                result[i][j] += valA * B[colA][j];
            }
        }
    }

    // Free CSR matrix
    free(csr_A->values);
    free(csr_A->col_indices);
    free(csr_A->row_start);
    free(csr_A);

    return result;
}