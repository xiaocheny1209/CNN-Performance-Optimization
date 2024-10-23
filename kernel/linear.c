#include "linear.h"
#include <stdio.h>

float *linear(float *input, float **weights, float *biases, int inputSize, int outputSize)
{
    float *output = (float *)malloc(outputSize * sizeof(float));
    for (int j = 0; j < outputSize; j++)
    {
        output[j] = biases[j];
        for (int i = 0; i < inputSize; i++)
        {
            output[j] += input[i] * weights[j][i];
        }
    }
    return output;
}
