#include <stdio.h>
#include "classifier.h"

FILE* fptr;

int main()
{
    fptr = fopen("model.mdl", "r");
    classifier_t* cls = init_classifier(fptr);
    if(cls == NULL)
    {
        return 1;
    }

    data_point_t test1 = {
        {-1, -1, -2, 4}
    };

    data_point_t test2 = {
        {20, -20, -20, 20}
    };

    printf("Test1 Class: %u\r\n Test2 Class: %u\r\n", classify(cls, &test1), classify(cls, &test2));

    destroy_classifier(cls);
    fclose(fptr);

    return 0;
}