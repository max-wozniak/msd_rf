#include <stdlib.h>
#include "preprocessor.h"

#define AVG_ORDER 3
#define BW_THRESH 23.0f
#define BW_MARGIN 20

#define PROC_WINDOW_W 25
#define PROC_WINDOW_C 255
#define PROC_WINDOW_L (PROC_WINDOW_C - PROC_WINDOW_W)
#define PROC_WINDOW_R (PROC_WINDOW_C + PROC_WINDOW_W)


float curr_avg[2048] = {0};
float prev_avg[2048] = {0};

void extract_features(
    data_point_t* o_features,
    float* i_curr_fft_buf, 
    float* i_prev_fft_buf, 
    uint32_t i_num_samples)
{
    uint32_t conv_samples = i_num_samples - AVG_ORDER + 1;
    float gain_max = 0.0f;
    uint8_t min_found = 0;
    float diff_max = 0.0f;
    float diff_curr;
    float gain_avg = 0.0f;
    float gain_var = 0.0f;
    float var_dif = 0.0;
    for (int i = 0; i < 2048; i++) {
    	curr_avg[i] = 0;
	prev_avg[i] = 0;
    }

    // Averaging and Diff
    for(uint8_t t = 0; t < AVG_ORDER; t++)
    {
        curr_avg[0] += i_curr_fft_buf[t] / (float)AVG_ORDER;
        prev_avg[0] += i_prev_fft_buf[t] / (float)AVG_ORDER;
    }

    for(uint32_t i = 1; i < conv_samples; i++)
    {
        for(uint8_t t = 0; t < AVG_ORDER; t++)
        {
            curr_avg[i] += i_curr_fft_buf[i + t] / (float)AVG_ORDER;
            prev_avg[i] += i_prev_fft_buf[i + t] / (float)AVG_ORDER;
        }
    }

    // Find max gain

    for (uint32_t i = PROC_WINDOW_L; i < PROC_WINDOW_R; i++)
    {
        if (curr_avg[i] > gain_max)
        {
            gain_max = curr_avg[i];
        }

        diff_curr = fabs((float)(curr_avg[i] - prev_avg[i]));
        if (diff_curr > diff_max)
        {
            diff_max = diff_curr;
        }
    }

    // Set features
    o_features->val[0] = gain_max;
    o_features->val[1] = diff_max;

}
