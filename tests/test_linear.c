#include "unity/unity.h"
#include "../kernel/kernel.h"
#include "test_linear.h"

const int REP_LIN = 100;

void test_linear_basic(void)
{
    float input[] = {1.0, 2.0, 3.0};
    float *weights[] = {(float[]){1.0, 2.0, 3.0}, (float[]){4.0, 5.0, 6.0}};
    float biases[] = {0.1, 0.2};
    float *output = linear(input, weights, biases, 3, 2);
    TEST_ASSERT_EQUAL_FLOAT(14.1, output[0]);
    TEST_ASSERT_EQUAL_FLOAT(32.2, output[1]);

    // Cleanup
    free(output);
}

void test_linear_basic2(void)
{
    float input[] = {1.0, 2.0, 3.0};
    float *weights[] = {(float[]){1.0, 2.0, 3.0}, (float[]){4.0, 5.0, 6.0}};
    float biases[] = {0.1, 0.2};
    float *output = linear(input, weights, biases, 3, 2);
    TEST_ASSERT_EQUAL_FLOAT(14.1, output[0]);
    TEST_ASSERT_EQUAL_FLOAT(32.2, output[1]);

    // Cleanup
    free(output);
}

// Add more test cases as needed

void test_linear_minimal(void)
{
    float input[] = {4.0};
    float *weights[] = {(float[]){3.0}};
    float biases[] = {1.0};
    float *output = linear(input, weights, biases, 1, 1);
    TEST_ASSERT_EQUAL_FLOAT(13.0, output[0]);
    // TEST_ASSERT_EQUAL_FLOAT(32.2, output[1]);

    // Cleanup
    free(output);
}

void test_linear_input_all_zeros(void)
{
    float input[] = {0.0, 0.0, 0.0};
    float *weights[] = {(float[]){1.0, 2.0, 3.0}, (float[]){4.0, 5.0, 6.0}};
    float biases[] = {0.1, 0.2};
    float *output = linear(input, weights, biases, 3, 2);
    TEST_ASSERT_EQUAL_FLOAT(0.1, output[0]);
    TEST_ASSERT_EQUAL_FLOAT(0.2, output[1]);

    // Cleanup
    free(output);
}

void test_linear_weight_all_zeros(void)
{
    float input[] = {1.0, 2.0, 3.0};
    float *weights[] = {(float[]){0.0, 0.0, 0.0}, (float[]){0.0, 0.0, 0.0}};
    float biases[] = {0.1, 0.2};
    float *output = linear(input, weights, biases, 3, 2);
    TEST_ASSERT_EQUAL_FLOAT(0.1, output[0]);
    TEST_ASSERT_EQUAL_FLOAT(0.2, output[1]);

    // Cleanup
    free(output);
}

void test_linear_medium_input(void)
{
    int inputSize = 128;
    int outputSize = 64;

    float *input = (float *)malloc(inputSize * sizeof(float));
    float **weights = (float **)malloc(outputSize * sizeof(float *));
    for (int i = 0; i < outputSize; i++)
    {
        weights[i] = (float *)malloc(inputSize * sizeof(float));
    }
    float *biases = (float *)malloc(outputSize * sizeof(float));

    for (int i = 0; i < inputSize; i++)
    {
        input[i] = (i % 2 == 0) ? i * 0.25f : -i * 0.25f;
    }

    for (int i = 0; i < outputSize; i++)
    {
        for (int j = 0; j < inputSize; j++)
        {
            weights[i][j] = (j % 4 == 0) ? j * 0.05f : -j * 0.05f;
        }
    }

    for (int i = 0; i < outputSize; i++)
    {
        biases[i] = (i % 2 == 0) ? i * 0.02f : -i * 0.02f;
    }

    float *result = linear(input, weights, biases, inputSize, outputSize);

    // check general properties, because the expected result cannot be manually computed
    for (int i = 0; i < outputSize; i++)
    {
        // Ensure the result is within reasonable range (e.g., not all zeros)
        TEST_ASSERT_NOT_EQUAL(0, result[i]);
        // Optionally check that values aren't too extreme (e.g., clamp test if necessary)
        TEST_ASSERT(result[i] > -10000.0f && result[i] < 10000.0f);
    }

    // Cleanup
    free(input);
    for (int i = 0; i < outputSize; i++)
    {
        free(weights[i]);
    }
    free(weights);
    free(biases);
    free(result);
}

void test_linear_large_input(void)
{
    int inputSize = 1024;
    int outputSize = 512;

    float *input = (float *)malloc(inputSize * sizeof(float));
    float **weights = (float **)malloc(outputSize * sizeof(float *));
    for (int i = 0; i < outputSize; i++)
    {
        weights[i] = (float *)malloc(inputSize * sizeof(float));
    }
    float *biases = (float *)malloc(outputSize * sizeof(float));

    // Initialize input with alternating positive and negative values
    for (int i = 0; i < inputSize; i++)
    {
        input[i] = (i % 2 == 0) ? i * 0.5f : -i * 0.5f;
    }

    // Initialize weights with random values and some negative values
    for (int i = 0; i < outputSize; i++)
    {
        for (int j = 0; j < inputSize; j++)
        {
            weights[i][j] = (j % 3 == 0) ? j * 0.1f : -j * 0.1f;
        }
    }

    // Initialize biases with a mix of positive and negative values
    for (int i = 0; i < outputSize; i++)
    {
        biases[i] = (i % 2 == 0) ? i * 0.01f : -i * 0.01f;
    }

    float *result = linear(input, weights, biases, inputSize, outputSize);

    // Expected result can't be manually pre-calculated here, so we can check general properties
    for (int i = 0; i < outputSize; i++)
    {
        // Since the weights and input are alternating positive/negative, result should be non-trivial
        TEST_ASSERT_NOT_EQUAL(0, result[i]);
    }

    // Cleanup
    free(input);
    for (int i = 0; i < outputSize; i++)
    {
        free(weights[i]);
    }
    free(weights);
    free(biases);
    free(result);
}

void profile_linear(int inputSize, int outputSize)
{
    float *input = (float *)malloc(inputSize * sizeof(float));
    float **weights = (float **)malloc(outputSize * sizeof(float *));
    for (int i = 0; i < outputSize; i++)
    {
        weights[i] = (float *)malloc(inputSize * sizeof(float));
    }
    float *biases = (float *)malloc(outputSize * sizeof(float));

    for (int i = 0; i < inputSize; i++)
    {
        input[i] = (float)(rand() % 10);
        ;
    }

    for (int i = 0; i < outputSize; i++)
    {
        for (int j = 0; j < inputSize; j++)
        {
            weights[i][j] = (float)(rand() % 10);
        }
    }

    for (int i = 0; i < outputSize; i++)
    {
        biases[i] = (float)(rand() % 10);
        ;
    }

    for (int r = 0; r < REP_LIN; r++)
    {
        float *result = linear(input, weights, biases, inputSize, outputSize);
        free(result);
    }

    free(input);
    for (int i = 0; i < outputSize; i++)
    {
        free(weights[i]);
    }
    free(weights);
    free(biases);
}