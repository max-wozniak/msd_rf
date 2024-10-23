#include <stdlib.h>
#include "classifier.h"

// Create classifier
classifier_t* init_classifier(uint8_t order)
{
    classifier_t* cls = (classifier_t*)malloc(sizeof(classifier_t));
    if(cls == NULL)
    {
        return NULL;
    }
    cls->order = order;
    cls->coeffs = (float*)malloc(NUM_FEATURES*order*sizeof(float));
    if(cls->coeffs == NULL)
    {
        return NULL;
    }
    return cls;
}

// Delete model
void destroy_classifier(classifier_t* cls)
{
    free(cls->coeffs);
    free(cls);
}

// Perform classification
uint8_t classify(classifier_t* cls, data_point_t* input)
{
    // Add the intercept
    float output = cls->coeffs[NUM_FEATURES];

    // Perform dot product between point and dividing hyperplane
    for(uint8_t xi = 0; xi < NUM_FEATURES; xi++)
    {
        float poly = 1.0;
        for(uint8_t order = 0; order < cls->order; order++)
        {
            poly *= input->val[xi];
            output += cls->coeffs[xi*NUM_FEATURES + order]*poly;
        }
    }

    output = (cls->positive_orientation > 0) ? output : output*(-1.0);

    // Get classification
    return (output > 0.0) ? 1 : 0;
}