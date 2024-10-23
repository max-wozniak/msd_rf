#include <stdint.h>

#define NUM_FEATURES 6

typedef struct {
    float val[NUM_FEATURES];
} data_point_t;

typedef struct {
    uint8_t order;
    uint8_t positive_orientation;
    float* coeffs;
} classifier_t;

uint8_t classify(classifier_t* cls, data_point_t* input);



