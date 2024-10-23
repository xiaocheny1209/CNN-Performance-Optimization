#include "functional.h"
#include <stdio.h>

float relu(float x)
{
    return (x >= 0) ? x : 0;
}

void applyRelu(float *input, int inputSize)
{
    for (int i = 0; i < inputSize; i++)
    {
        input[i] = relu(input[i]);
    }
}

float *softmax(float *input, int inputSize)
{
    float *output = (float *)malloc(inputSize * sizeof(float));

    // Find maximum of input vector
    float maxInput = input[0];
    for (int i = 1; i < inputSize; i++)
    {
        if (input[i] > maxInput)
        {
            maxInput = input[i];
        }
    }

    // Compute exp of input - maxInput to avoid underflow
    float sumExp = 0.0f;
    for (int i = 0; i < inputSize; i++)
    {
        output[i] = exp(input[i] - maxInput);
        sumExp += output[i];
    }

    // Normalise and apply log
    for (int i = 0; i < inputSize; i++)
    {
        output[i] = log(output[i] / sumExp); // log softmax
    }

    return output;
}