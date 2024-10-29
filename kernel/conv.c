#include "conv.h"
#include "functional.h"
#include "matrix_ops.h"
#include <stdio.h>

// Basic convolution operation
float ***convolution(float ***image, int numChannels, float ****kernel, float *biasData, int numFilters, int inputSize, int kernelSize)
{
    int outputSize = inputSize - kernelSize + 1; // stride = 1 and no padding
    float ***result = (float ***)malloc(numFilters * sizeof(float **));
    for (int f = 0; f < numFilters; f++)
    {
        result[f] = (float **)malloc(outputSize * sizeof(float *));
        for (int i = 0; i < outputSize; i++)
        {
            result[f][i] = (float *)malloc(outputSize * sizeof(float));
            for (int j = 0; j < outputSize; j++)
            {
                result[f][i][j] = 0;
            }
        }
    }

    for (int f = 0; f < numFilters; f++)
    {
        for (int c = 0; c < numChannels; c++)
        {
            for (int i = 0; i < outputSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    // Extract the patch from the image (kernelSize x kernelSize)
                    float **image_patch = (float **)malloc(kernelSize * sizeof(float *));
                    for (int ki = 0; ki < kernelSize; ki++)
                    {
                        image_patch[ki] = (float *)malloc(kernelSize * sizeof(float));
                        for (int kj = 0; kj < kernelSize; kj++)
                        {
                            image_patch[ki][kj] = image[c][i + ki][j + kj];
                        }
                    }

                    float **conv_result = (float **)malloc(kernelSize * sizeof(float *));
                    for (int i = 0; i < kernelSize; i++)
                    {
                        conv_result[i] = (float *)malloc(kernelSize * sizeof(float));
                    }
                    for (int i = 0; i < kernelSize; i++)
                    {
                        for (int j = 0; j < kernelSize; j++)
                        {
                            conv_result[i][j] = image_patch[i][j] * kernel[f][c][i][j];
                        }
                    }

                    // float **conv_result = matmul(image_patch, kernel[f][c], kernelSize, kernelSize, kernelSize, kernelSize);

                    for (int ki = 0; ki < kernelSize; ki++)
                    {
                        for (int kj = 0; kj < kernelSize; kj++)
                        {
                            result[f][i][j] += conv_result[ki][kj];
                        }
                    }

                    for (int ki = 0; ki < kernelSize; ki++)
                    {
                        free(image_patch[ki]);
                        free(conv_result[ki]);
                    }
                    free(image_patch);
                    free(conv_result);
                }
            }
        }
        // Apply the bias for filter `f`
        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                result[f][i][j] += biasData[f];
            }
        }

        // Apply ReLU activation function for each output of filter `f`
        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                result[f][i][j] = relu(result[f][i][j]);
            }
        }
    }

    return result;
}
