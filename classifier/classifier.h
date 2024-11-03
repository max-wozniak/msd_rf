#include <stdint.h>
#include <stdio.h>

#define NUM_FEATURES 4

typedef struct {
    float val[NUM_FEATURES];
} data_point_t;

typedef struct {
    uint8_t order;
    int polarity;
    float* coeffs;
} classifier_t;

uint8_t classify(classifier_t* cls, data_point_t* input);
classifier_t* init_classifier(FILE* fptr);
void destroy_classifier(classifier_t* cls);



