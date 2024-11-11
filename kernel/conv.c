#include "conv.h"
#include "functional.h"
#include "matrix_ops.h"
#include <stdio.h>

// Im2col algorithm: flatten an image to a 1d vector
float **im2col(float ***image, int numChannels, int imageSize, int kernelSize, int stride, int *outputSize)
{
    // Calculate the output size (height and width) after applying the kernel
    *outputSize = (imageSize - kernelSize) / stride + 1;

    // Allocate memory for the output matrix
    // The output matrix has dimensions [numChannels * kernelSize * kernelSize, outputSize * outputSize]
    int matrixHeight = numChannels * kernelSize * kernelSize;
    int matrixWidth = (*outputSize) * (*outputSize);
    float **outputMatrix = (float **)malloc(matrixHeight * sizeof(float *));
    for (int i = 0; i < matrixHeight; i++)
    {
        outputMatrix[i] = (float *)malloc(matrixWidth * sizeof(float));
    }

    // Convert image into columns
    int colIndex = 0;
    for (int c = 0; c < numChannels; c++)
    { // Iterate over each channel
        for (int row = 0; row <= imageSize - kernelSize; row += stride)
        { // Sliding over image rows
            for (int col = 0; col <= imageSize - kernelSize; col += stride)
            { // Sliding over image columns
                // Flatten the kernel-sized patch into the output matrix
                int patchIndex = 0;
                for (int i = 0; i < kernelSize; i++)
                {
                    for (int j = 0; j < kernelSize; j++)
                    {
                        outputMatrix[patchIndex++][colIndex] = image[c][row + i][col + j];
                    }
                }
                colIndex++;
            }
        }
    }

    return outputMatrix;
}

// Im2col algorithm's inverse: convert the 1d vector back to image
float ***col2im(float **result, int num_kernels, int conv_rows, int conv_cols)
{
    // Allocate memory for the reconstructed 3D image
    float ***output = (float ***)malloc(num_kernels * sizeof(float **));
    for (int k = 0; k < num_kernels; k++)
    {
        output[k] = (float **)malloc(conv_rows * sizeof(float *));
        for (int i = 0; i < conv_rows; i++)
        {
            output[k][i] = (float *)calloc(conv_cols, sizeof(float));
        }
    }

    // Map each element from result back to the output image
    int col = 0;
    for (int k = 0; k < num_kernels; k++)
    {
        for (int i = 0; i < conv_rows; i++)
        {
            for (int j = 0; j < conv_cols; j++)
            {
                for (int ki = 0; ki < num_kernels; ki++)
                {
                    for (int kj = 0; kj < num_kernels; kj++)
                    {
                        int row_idx = i + ki;
                        int col_idx = j + kj;

                        if (row_idx < conv_rows && col_idx < conv_cols)
                        {
                            output[k][row_idx][col_idx] += result[k][col++];
                        }
                    }
                }
            }
        }
    }

    return output;
}

float **kernel_flatten(float ****kernel, int num_kernels, int kernel_size)
{
    float **flattened_kernels = (float **)malloc(num_kernels * sizeof(float *));
    for (int i = 0; i < num_kernels; i++)
    {
        flattened_kernels[i] = (float *)malloc(kernel_size * kernel_size * sizeof(float));
    }

    for (int k = 0; k < num_kernels; k++)
    {
        for (int i = 0; i < kernel_size; i++)
        {
            for (int j = 0; j < kernel_size; j++)
            {
                flattened_kernels[k][i * kernel_size + j] = kernel[k][0][i][j];
            }
        }
    }

    return flattened_kernels;
}

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

void print_2d_array(float **array, int height, int width)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            printf("%d ", (int)array[i][j]);
        }
        printf("\n");
    }
}

// Convolution with im2col algorithm: im2col first and finally convert the results back to image with col2im
float ***convolution_im2col(float ***image, int numChannels, float ****kernel, float *biasData, int numFilters, int inputSize, int kernelSize, MatmulType matmul_type)
{
    // Flatten kernel
    float **flattened_kernel = kernel_flatten(kernel, numFilters, kernelSize);

    // Apply im2col
    int outputSize;
    float **im2col_output = im2col(image, numChannels, inputSize, kernelSize, 1, &outputSize);

    // Apply matmul
    float **conv_output;
    if (matmul_type == MATMUL_SPARSE)
    {
        conv_output = matmul_sparse(flattened_kernel, im2col_output, numFilters, numChannels * kernelSize * kernelSize, numChannels * kernelSize * kernelSize, outputSize * outputSize);
    }
    else
    { // MATMUL_BASE
        conv_output = matmul(flattened_kernel, im2col_output, numFilters, numChannels * kernelSize * kernelSize, numChannels * kernelSize * kernelSize, outputSize * outputSize);
    }

    // Apply col2im
    float ***output_image = col2im(conv_output, numFilters, outputSize, outputSize);

    // Add bias and apply ReLU
    for (int f = 0; f < numFilters; f++)
    {
        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                // Add bias
                output_image[f][i][j] += biasData[f];
                // Apply ReLU activation function
                output_image[f][i][j] = relu(output_image[f][i][j]);
            }
        }
    }

    // Cleanup
    for (int i = 0; i < numFilters; i++)
    {
        free(flattened_kernel[i]);
    }
    free(flattened_kernel);

    for (int i = 0; i < numChannels * kernelSize * kernelSize; i++)
    {
        free(im2col_output[i]);
    }
    free(im2col_output);

    for (int i = 0; i < numFilters; i++)
    {
        free(conv_output[i]);
    }
    free(conv_output);

    return output_image;
}