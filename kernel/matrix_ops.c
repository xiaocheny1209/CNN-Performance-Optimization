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