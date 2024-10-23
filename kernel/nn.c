#include "nn.h"
#include <stdio.h>

float *flatten(float ***input, int inputSize, int depth)
{
    int flatSize = inputSize * inputSize * depth;
    float *flatOutput = (float *)malloc(flatSize * sizeof(float));

    int index = 0;
    for (int d = 0; d < depth; d++)
    {
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                flatOutput[index++] = input[d][i][j]; // Flatten the 3D array into 1D
            }
        }
    }

    return flatOutput;
}

void destroyConvOutput(float ***convOutput, int convOutputSize)
{
    for (int i = 0; i < 32; i++)
    {
        for (int j = 0; j < convOutputSize; j++)
        {
            free(convOutput[i][j]);
        }
        free(convOutput[i]);
    }
    free(convOutput);
}

int forwardPass(float ***image, int numChannels, float ****conv1WeightsData, float **fc1WeightsData, float **fc2WeightsData, float *conv1BiasData, float *fc1BiasData, float *fc2BiasData)
{
    // 1. Perform the convolution operation

    // 2. Flatten the output

    // 3. Perform the fully connected operations

    // 4. Apply the final softmax activation

    // 5. Make predictions

    // Clean up the memory usage
    return 0;
}

int predict(float *probabilityVector, int numClasses)
{
    int bestClass = 0;
    float maxProbability = probabilityVector[0];

    for (int i = 1; i < numClasses; i++)
    {
        if (probabilityVector[i] > maxProbability)
        {
            bestClass = i;
            maxProbability = probabilityVector[i];
        }
    }

    return bestClass;
}
