#include "unity/unity.h"
#include "test_conv.h"
#include "test_nn.h"
#include "test_functional.h"
#include "test_linear.h"
#include "test_matrix_ops.h"
#include "test_attention.h"
#include <stdio.h>

const int SMALL_MAT = 8;
const int MEDIUM_MAT = 256;
const int LARGE_MAT = 512;

void setUp(void)
{
    /* Code here will run before each test */
}

void tearDown(void)
{
    /* Code here will run after each test */
}

int main(void)
{
    // profiling
    // profile_linear(SMALL_MAT, SMALL_MAT);
    // profile_matmul(SMALL_MAT);
    // profile_matmul(MEDIUM_MAT);
    // profile_matmul(LARGE_MAT);
    // profile_conv(SMALL_MAT);
    // profile_softmax_small_input();
    // profile_softmax_medium_input();
    // profile_softmax_large_input();
    // profile_relu_small_input();
    // profile_relu_medium_input();
    // profile_relu_large_input();

    // tests

    // UNITY_BEGIN();

    // Test conv
    RUN_TEST(test_conv);
    RUN_TEST(test_conv_minimal);
    RUN_TEST(test_conv_multiple_filters_channels);

    // Test nn
    // RUN_TEST(test_flatten_basic);
    // RUN_TEST(test_predict_simple_array);
    // RUN_TEST(test_predict_all_same_values);
    // RUN_TEST(test_predict_mix_of_negatives_and_positives);

    // Test functional
    // RUN_TEST(test_softmax_basic);
    // RUN_TEST(test_relu);

    // Test linear
    // RUN_TEST(test_linear_basic);
    // RUN_TEST(test_linear_basic2);
    // RUN_TEST(test_linear_minimal);
    // RUN_TEST(test_linear_input_all_zeros);
    // RUN_TEST(test_linear_weight_all_zeros);
    // RUN_TEST(test_linear_medium_input);
    // RUN_TEST(test_linear_large_input);

    // Test matrix_ops
    // RUN_TEST(test_matmul_square_matrices);
    // RUN_TEST(test_matmul_incompatible_dimensions);
    // RUN_TEST(test_matmul_minimal);
    // RUN_TEST(test_matmul_with_zeros);
    // RUN_TEST(test_matmul_with_negatives);

    // RUN_TEST(test_scaled_dot_product_attention);

    // return UNITY_END();
}
