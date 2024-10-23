#include "unity/unity.h"
#include "../kernel/kernel.h"
#include "test_functional.h"
#include <float.h>

const int SMALL = 200;
const int MEDIUM = 4000;
const int LARGE = 50000;

void test_softmax_basic(void)
{
    float input[] = {1.0, 2.0, 3.0};
    float *output = softmax(input, 3);
    float sum = 0.0;

    // Check that the sum of the output is 0 (because the output is log softmax)
    for (int i = 0; i < 3; i++)
    {
        sum += expf(output[i]);
    }

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0, sum);

    // Check that the maximum input corresponds to the maximum output
    int maxInputIndex = 0;
    int maxOutputIndex = 0;

    for (int i = 1; i < 3; i++)
    {
        if (input[i] > input[maxInputIndex])
        {
            maxInputIndex = i;
        }
        if (output[i] > output[maxOutputIndex])
        {
            maxOutputIndex = i;
        }
    }

    TEST_ASSERT_EQUAL_INT(maxInputIndex, maxOutputIndex);

    // Cleanup
    free(output);
}

void profile_softmax_small_input(void)
{
    int size = SMALL;
    float input[size];
    for (int i = 0; i < size; i++)
    {
        input[i] = rand() % 10;
    }
    float *output = softmax(input, size);
}

void profile_softmax_medium_input(void)
{
    int size = MEDIUM;
    float input[size];
    for (int i = 0; i < size; i++)
    {
        input[i] = rand() % 10;
    }
    float *output = softmax(input, size);
}

void profile_softmax_large_input(void)
{
    int size = LARGE;
    float input[size];
    for (int i = 0; i < size; i++)
    {
        input[i] = rand() % 10;
    }
    float *output = softmax(input, size);
}

void test_relu(void)
{
    float inputs[] = {3.0f, 0.0f, -3.0f};
    float expected_outputs[] = {3.0f, 0.0f, 0.0f};
    int test_cases = sizeof(inputs) / sizeof(inputs[0]);

    for (int i = 0; i < test_cases; i++)
    {
        float output = relu(inputs[i]);
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected_outputs[i], output);
    }
}

void profile_relu_small_input(void)
{
    int size = SMALL;
    float input[size];
    for (int i = 0; i < size; i++)
    {
        input[i] = rand() % 10;
    }
    applyRelu(input, size);
}

void profile_relu_medium_input(void)
{
    int size = MEDIUM;
    float input[size];
    for (int i = 0; i < size; i++)
    {
        input[i] = rand() % 10;
    }
    applyRelu(input, size);
}

void profile_relu_large_input(void)
{
    int size = LARGE;
    float input[size];
    for (int i = 0; i < size; i++)
    {
        input[i] = rand() % 10;
    }
    applyRelu(input, size);
}