#include <stdlib.h>
#include <string.h>
#include "classifier.h"


#define SBUF_SIZE 1024
const char* delim = " ";

// Create classifier
classifier_t* init_classifier(FILE* fptr)
{
    char fstr[SBUF_SIZE];
    char* token;

    fgets(fstr, SBUF_SIZE, fptr);

    classifier_t* cls = (classifier_t*)malloc(sizeof(classifier_t));
    if(cls == NULL)
    {
        return NULL;
    }

    // Get the first token
    token = strtok(fstr, delim);
    cls->order = (uint8_t)atoi(token);

    cls->coeffs = (float*)calloc(NUM_FEATURES*cls->order + 1, sizeof(float));
    if(cls->coeffs == NULL)
    {
        return NULL;
    }
    
    token = strtok(NULL, delim);
    cls->polarity = atoi(token);

    token = strtok(NULL, delim);
    // Continue processing tokens until the end of the string
    uint8_t i = 0;
    while (token != NULL) {
        // Convert the token to a float
        cls->coeffs[i] = atof(token);

        // Get the next token
        token = strtok(NULL, delim);
        i++;
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

    output *= cls->polarity;

    // Get classification
    return (output > 0.0) ? 1 : 0;
}