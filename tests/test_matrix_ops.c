#include "unity/unity.h"
#include "../kernel/kernel.h"
#include "test_matrix_ops.h"

#define EPSILON 0.000001f

const int REP = 10;

void assert_float_array_equal_matmul(float **expected, float **actual, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            UNITY_TEST_ASSERT_FLOAT_WITHIN(EPSILON, expected[i][j], actual[i][j], __LINE__, "Arrays Not Equal!");
        }
    }
}

void test_matmul_square_matrices(void)
{
    // Setup
    float **A = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++)
    {
        A[i] = (float *)malloc(2 * sizeof(float));
    }
    A[0][0] = 1.0f;
    A[0][1] = 2.0f;
    A[1][0] = 3.0f;
    A[1][1] = 4.0f;

    float **B = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++)
    {
        B[i] = (float *)malloc(2 * sizeof(float));
    }
    B[0][0] = 2.0f;
    B[0][1] = 0.0f;
    B[1][0] = 1.0f;
    B[1][1] = 2.0f;

    float **expected = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++)
    {
        expected[i] = (float *)malloc(2 * sizeof(float));
    }
    expected[0][0] = 4.0f;
    expected[0][1] = 4.0f;
    expected[1][0] = 10.0f;
    expected[1][1] = 8.0f;

    // Run function under test
    float **C = matmul(A, B, 2, 2, 2, 2);

    // Check expectations
    assert_float_array_equal_matmul(expected, C, 2, 2);

    // Cleanup
    for (int i = 0; i < 2; i++)
    {
        free(A[i]);
        free(B[i]);
        free(C[i]);
        free(expected[i]);
    }
    free(A);
    free(B);
    free(C);
    free(expected);
}

void test_matmul_incompatible_dimensions(void)
{
    // Setup
    float **A = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++)
    {
        A[i] = (float *)malloc(3 * sizeof(float));
    }

    float **B = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++)
    {
        B[i] = (float *)malloc(2 * sizeof(float));
    }

    // Run function under test
    float **C = matmul(A, B, 2, 3, 2, 2);

    // Check expectations
    UNITY_TEST_ASSERT_NULL(C, __LINE__, "Expected NULL!");

    // Cleanup
    for (int i = 0; i < 2; i++)
    {
        free(A[i]);
        free(B[i]);
    }
    free(A);
    free(B);
}

void test_matmul_minimal(void)
{
    float **A = (float **)malloc(1 * sizeof(float *));
    for (int i = 0; i < 1; i++)
    {
        A[i] = (float *)malloc(1 * sizeof(float));
    }
    A[0][0] = 1.0f;

    float **B = (float **)malloc(1 * sizeof(float *));
    for (int i = 0; i < 1; i++)
    {
        B[i] = (float *)malloc(1 * sizeof(float));
    }
    B[0][0] = 2.0f;

    float **expected = (float **)malloc(1 * sizeof(float *));
    for (int i = 0; i < 1; i++)
    {
        expected[i] = (float *)malloc(1 * sizeof(float));
    }
    expected[0][0] = 2.0f;

    float **C = matmul(A, B, 1, 1, 1, 1);

    assert_float_array_equal_matmul(expected, C, 1, 1);

    for (int i = 0; i < 1; i++)
    {
        free(A[i]);
        free(B[i]);
        free(C[i]);
        free(expected[i]);
    }
    free(A);
    free(B);
    free(C);
    free(expected);
}

void test_matmul_with_zeros(void)
{
    float **A = (float **)malloc(3 * sizeof(float *));
    for (int i = 0; i < 3; i++)
    {
        A[i] = (float *)malloc(3 * sizeof(float));
    }
    A[0][0] = 1.0f;
    A[0][1] = 0.0f;
    A[0][2] = 2.0f;
    A[1][0] = 0.0f;
    A[1][1] = 0.0f;
    A[1][2] = 0.0f;
    A[2][0] = 3.0f;
    A[2][1] = 4.0f;
    A[2][2] = 0.0f;

    float **B = (float **)malloc(3 * sizeof(float *));
    for (int i = 0; i < 3; i++)
    {
        B[i] = (float *)malloc(3 * sizeof(float));
    }
    B[0][0] = 0.0f;
    B[0][1] = 1.0f;
    B[0][2] = 0.0f;
    B[1][0] = 2.0f;
    B[1][1] = 0.0f;
    B[1][2] = 0.0f;
    B[2][0] = 0.0f;
    B[2][1] = 3.0f;
    B[2][2] = 4.0f;

    float **expected = (float **)malloc(3 * sizeof(float *));
    for (int i = 0; i < 3; i++)
    {
        expected[i] = (float *)malloc(3 * sizeof(float));
    }
    expected[0][0] = 0.0f;
    expected[0][1] = 7.0f;
    expected[0][2] = 8.0f;
    expected[1][0] = 0.0f;
    expected[1][1] = 0.0f;
    expected[1][2] = 0.0f;
    expected[2][0] = 8.0f;
    expected[2][1] = 3.0f;
    expected[2][2] = 0.0f;

    float **C = matmul(A, B, 3, 3, 3, 3);
    assert_float_array_equal_matmul(expected, C, 3, 3);

    for (int i = 0; i < 3; i++)
    {
        free(A[i]);
        free(B[i]);
        free(C[i]);
        free(expected[i]);
    }
    free(A);
    free(B);
    free(C);
    free(expected);
}

void test_matmul_with_negatives(void)
{
    float **A = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++)
    {
        A[i] = (float *)malloc(2 * sizeof(float));
    }
    A[0][0] = -1.0f;
    A[0][1] = 2.0f;
    A[1][0] = 3.0f;
    A[1][1] = -4.0f;

    float **B = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++)
    {
        B[i] = (float *)malloc(2 * sizeof(float));
    }
    B[0][0] = 5.0f;
    B[0][1] = -6.0f;
    B[1][0] = -7.0f;
    B[1][1] = 8.0f;

    float **expected = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++)
    {
        expected[i] = (float *)malloc(2 * sizeof(float));
    }
    expected[0][0] = -19.0f;
    expected[0][1] = 22.0f;
    expected[1][0] = 43.0f;
    expected[1][1] = -50.0f;

    float **C = matmul(A, B, 2, 2, 2, 2);
    assert_float_array_equal_matmul(expected, C, 2, 2);

    // Cleanup
    for (int i = 0; i < 2; i++)
    {
        free(A[i]);
        free(B[i]);
        free(C[i]);
        free(expected[i]);
    }
    free(A);
    free(B);
    free(C);
    free(expected);
}

void profile_matmul(int size)
{
    float **A = (float **)malloc(size * sizeof(float *));
    float **B = (float **)malloc(size * sizeof(float *));
    for (int i = 0; i < size; i++)
    {
        A[i] = (float *)malloc(size * sizeof(float));
        B[i] = (float *)malloc(size * sizeof(float));
        for (int j = 0; j < size; j++)
        {
            A[i][j] = (float)(rand() % 10);
            B[i][j] = (float)(rand() % 10);
        }
    }

    // Profiling: run the function multiple times to minimize setup/cleanup impact
    for (int r = 0; r < REP; r++)
    {
        float **result = matmul(A, B, size, size, size, size);

        for (int i = 0; i < size; i++)
        {
            free(result[i]);
        }
        free(result);
    }

    for (int i = 0; i < size; i++)
    {
        free(A[i]);
        free(B[i]);
    }
    free(A);
    free(B);
}