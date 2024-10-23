#ifndef TEST_FUNCTIONAL_H
#define TEST_FUNCTIONAL_H

void test_softmax_basic(void);
void test_relu(void);

void profile_softmax_small_input(void);
void profile_softmax_medium_input(void);
void profile_softmax_large_input(void);
void profile_relu_small_input(void);
void profile_relu_medium_input(void);
void profile_relu_large_input(void);

#endif /* TEST_FUNCTIONAL_H */
