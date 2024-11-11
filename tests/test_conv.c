#include "unity/unity.h"
#include "../kernel/kernel.h"
#include "../utils/data_utils.h"
#include "test_conv.h"

#define EPSILON 0.000001f

const int REP_CONV = 100;

void print_3d_array(float ***array, int numFilters, int height, int width)
{
    for (int f = 0; f < numFilters; f++)
    {
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                printf("%d ", (int)array[f][i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

// newly added: allocate memory for the array
float ***create_image(int numChannels, int inputSize)
{
    float ***image = (float ***)malloc(numChannels * sizeof(float **));
    for (int c = 0; c < numChannels; c++)
    {
        image[c] = (float **)malloc(inputSize * sizeof(float *));
        for (int h = 0; h < inputSize; h++)
        {
            image[c][h] = (float *)malloc(inputSize * sizeof(float));
        }
    }
    return image;
}

void free_image(float ***image, int numChannels, int inputSize)
{
    for (int c = 0; c < numChannels; c++)
    {
        for (int i = 0; i < inputSize; i++)
        {
            free(image[c][i]);
        }
        free(image[c]);
    }
    free(image);
}

// newly added: allocate memory for the array
float ****create_kernel(int numFilters, int numChannels, int kernelSize)
{
    float ****kernel = (float ****)malloc(numFilters * sizeof(float ***));
    for (int f = 0; f < numFilters; f++)
    {
        kernel[f] = (float ***)malloc(numChannels * sizeof(float **));
        for (int c = 0; c < numChannels; c++)
        {
            kernel[f][c] = (float **)malloc(kernelSize * sizeof(float *));
            for (int h = 0; h < kernelSize; h++)
            {
                kernel[f][c][h] = (float *)malloc(kernelSize * sizeof(float));
            }
        }
    }
    return kernel;
}

void free_kernel(float ****kernel, int numFilters, int numChannels, int kernelSize)
{
    for (int i = 0; i < numFilters; i++)
    {
        for (int j = 0; j < numChannels; j++)
        {
            for (int k = 0; k < kernelSize; k++)
            {
                free(kernel[i][j][k]);
            }
            free(kernel[i][j]);
        }
        free(kernel[i]);
    }
    free(kernel);
}

void free_convOutput(float ***convOutput, int numFilters, int outputSize)
{
    for (int i = 0; i < numFilters; i++)
    {
        for (int j = 0; j < outputSize; j++)
        {
            free(convOutput[i][j]);
        }
        free(convOutput[i]);
    }
    free(convOutput);
}

void assert_float_array_equal_conv(float ***expected, float ***actual, int depth, int rows, int cols)
{
    for (int i = 0; i < depth; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            for (int k = 0; k < cols; k++)
            {
                UNITY_TEST_ASSERT_FLOAT_WITHIN(EPSILON, expected[i][j][k], actual[i][j][k], __LINE__, "Arrays Not Equal!");
            }
        }
    }
}

void test_conv(void)
{
    // Setup
    float image_data[1][5][5] = {
        {{0, 0, 0, 0, 0},
         {0, 1, 1, 1, 0},
         {0, 1, 0, 1, 0},
         {0, 1, 1, 1, 0},
         {0, 0, 0, 0, 0}}};
    int numChannels = 1;
    float ***image = init_image(image_data, 5, numChannels);

    float kernel_data[1][1][3][3] = {
        {{{1, 0, 1},
          {2, 0, 2},
          {1, 0, 1}}}};
    int numFilters = 1;
    int kernelSize = 3;
    float ****kernel = init_kernel(kernel_data, numFilters, kernelSize);

    // Initialize the bias
    float *biasData = (float *)malloc(1 * sizeof(float));
    biasData[0] = 0;

    float ***expected = (float ***)malloc(1 * sizeof(float **));
    for (int i = 0; i < 1; i++)
    {
        expected[i] = (float **)malloc(3 * sizeof(float *));
        for (int j = 0; j < 3; j++)
        {
            expected[i][j] = (float *)malloc(3 * sizeof(float));
        }
    }

    // Initialize the expected result
    float expected_values[1][3][3] = {
        {{2, 6, 2},
         {2, 8, 2},
         {2, 6, 2}}};
    for (int i = 0; i < 1; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                expected[i][j][k] = expected_values[i][j][k];
            }
        }
    }

    // Run function under test
    // float ***convOutput = convolution(image, numChannels, kernel, biasData, 1, 5, 3);
    float ***convOutput = convolution_im2col(image, numChannels, kernel, biasData, 1, 5, 3, MATMUL_SPARSE);
    // printf("final output ready for testing\n");

    // Check expectations
    assert_float_array_equal_conv(expected, convOutput, 1, 3, 3);

    // Cleanup
    free_image(image, numChannels, 5);
    free_kernel(kernel, 1, 1, 3);
    free(biasData);
    for (int i = 0; i < 1; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            free(expected[i][j]);
        }
        free(expected[i]);
    }
    free(expected);
    free_convOutput(convOutput, numFilters, kernelSize);
}

void test_conv_minimal(void)
{
    float ***image_data = (float ***)malloc(1 * sizeof(float **));
    image_data[0] = (float **)malloc(1 * sizeof(float *));
    image_data[0][0] = (float *)malloc(1 * sizeof(float));
    image_data[0][0][0] = 1;

    float ****kernel_data = (float ****)malloc(1 * sizeof(float ***));
    kernel_data[0] = (float ***)malloc(1 * sizeof(float **));
    kernel_data[0][0] = (float **)malloc(1 * sizeof(float *));
    kernel_data[0][0][0] = (float *)malloc(1 * sizeof(float));
    kernel_data[0][0][0][0] = 1;

    float *biasData = (float *)malloc(1 * sizeof(float));
    biasData[0] = 0;

    float ***expected = (float ***)malloc(1 * sizeof(float **));
    expected[0] = (float **)malloc(1 * sizeof(float *));
    expected[0][0] = (float *)malloc(1 * sizeof(float));
    expected[0][0][0] = 1; // expected result of 1x1 conv

    float ***convOutput = convolution(image_data, 1, kernel_data, biasData, 1, 1, 1);

    assert_float_array_equal_conv(expected, convOutput, 1, 1, 1);

    free_image(image_data, 1, 1);
    free_kernel(kernel_data, 1, 1, 1);
    free(biasData);
    free(expected[0][0]);
    free(expected[0]);
    free(expected);
    free_convOutput(convOutput, 1, 1);
}

void test_conv_multiple_filters_channels(void)
{
    // Image with 2 channels, 5x5 size
    float image_data[2][5][5] = {
        {// Channel 1
         {0, 0, -1, 0, 0},
         {0, 1, -2, 1, 0},
         {-1, -2, 0, 2, 1},
         {0, 1, 2, 1, 0},
         {0, 0, -1, 0, 0}},
        {// Channel 2
         {1, 0, -1, 0, 1},
         {0, -1, -2, -1, 0},
         {-1, -2, 0, 2, 1},
         {0, 1, 2, 1, 0},
         {1, 0, -1, 0, 1}}};
    int numChannels = 2;
    float ***image = init_image(image_data, 5, numChannels);

    // 2 filters, each with 2 channels, 3x3 kernel
    float kernel_data[2][2][3][3] = {
        { // Filter 1
         {// Channel 1 of filter 1
          {1, 0, -1},
          {2, 0, -2},
          {1, 0, -1}},
         {// Channel 2 of filter 1
          {-1, 0, 1},
          {2, 0, -2},
          {-1, 0, 1}}},
        { // Filter 2
         {// Channel 1 of filter 2
          {-1, 1, 0},
          {0, -2, 1},
          {1, 0, -1}},
         {// Channel 2 of filter 2
          {0, 1, -1},
          {-1, 2, 0},
          {1, -1, 0}}}};
    int numFilters = 2;
    int kernelSize = 3;
    float ****kernel = create_kernel(numFilters, numChannels, kernelSize);

    for (int f = 0; f < numFilters; f++)
    {
        for (int c = 0; c < numChannels; c++)
        {
            for (int h = 0; h < kernelSize; h++)
            {
                for (int w = 0; w < kernelSize; w++)
                {
                    kernel[f][c][h][w] = kernel_data[f][c][h][w];
                }
            }
        }
    }

    // Initialize the bias for both filters
    float *biasData = (float *)malloc(2 * sizeof(float));
    biasData[0] = 0; // Filter 1 bias
    biasData[1] = 1; // Filter 2 bias

    // Expected result after convolution
    float expected_values[2][3][3] = {
        {// Filter 1
         {7, 0, 0},
         {0, 0, 0},
         {0, 0, 9}},
        {// Filter 2
         {0, 0, 0},
         {1, 0, 7},
         {2, 2, 0}}};

    float ***expected = (float ***)malloc(2 * sizeof(float **));
    for (int f = 0; f < 2; f++)
    {
        expected[f] = (float **)malloc(3 * sizeof(float *));
        for (int i = 0; i < 3; i++)
        {
            expected[f][i] = (float *)malloc(3 * sizeof(float));
            for (int j = 0; j < 3; j++)
            {
                expected[f][i][j] = expected_values[f][i][j];
            }
        }
    }

    // Run function under test
    float ***convOutput = convolution(image, numChannels, kernel, biasData, numFilters, 5, 3);

    // print_3d_array(convOutput, numFilters, 3, 3);

    // Check expectations
    assert_float_array_equal_conv(expected, convOutput, numFilters, 3, 3);

    // Cleanup
    free_image(image, numChannels, 5);
    free_kernel(kernel, numFilters, numChannels, kernelSize);
    free(biasData);
    free_convOutput(convOutput, numFilters, 3);
}

void profile_conv(int inputSize)
{
    int numChannels = 1;
    float image_data[1][inputSize][inputSize];
    for (int i = 0; i < inputSize; i++)
    {
        for (int j = 0; j < inputSize; j++)
        {
            image_data[0][i][j] = (float)(rand() % 10);
        }
    }
    float ***image = create_image(numChannels, inputSize);

    float kernel_data[1][1][3][3] = {
        {{{1, 0, 1},
          {2, 0, 2},
          {1, 0, 1}}}};
    int numFilters = 1;
    int kernelSize = 3;
    float ****kernel = init_kernel(kernel_data, numFilters, kernelSize);

    float *biasData = (float *)malloc(1 * sizeof(float));
    biasData[0] = 0;

    for (int r = 0; r < REP_CONV; r++)
    {
        float ***convOutput = convolution(image, numChannels, kernel, biasData, 1, 5, 3);
        free_convOutput(convOutput, numFilters, kernelSize);
    }

    free_image(image, numChannels, inputSize);
    free_kernel(kernel, 1, 1, 3);
    free(biasData);
}