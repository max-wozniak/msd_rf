#include <stdint.h>

#define NUM_FEATURES 4

typedef struct {
    float val[NUM_FEATURES];
} data_point_t;

typedef struct {
    uint8_t order;
    uint8_t positive_orientation;
    float* coeffs;
} classifier_t;

uint8_t classify(classifier_t* cls, data_point_t* input);
classifier_t* init_classifier(uint8_t order, uint8_t orient);
void destroy_classifier(classifier_t* cls);



