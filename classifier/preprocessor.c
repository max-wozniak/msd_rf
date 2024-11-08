#include <stdlib.h>
#include "preprocessor.h"

#define AVG_ORDER 12
#define BW_THRESH 23.0f
#define BW_MARGIN 20

float curr_avg[2048] = {0};
float prev_avg[2048] = {0};

void extract_features(
    data_point_t* o_features,
    float* i_curr_fft_buf, 
    float* i_prev_fft_buf, 
    uint32_t i_num_samples)
{
    uint32_t conv_samples = i_num_samples - AVG_ORDER + 1;
    uint32_t fmin = 0;
    uint32_t fmax = i_num_samples-1;
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
    diff_max = curr_avg[0] - prev_avg[0];
    if (curr_avg[0] > BW_THRESH) min_found = 1;

    for(uint32_t i = 1; i < conv_samples; i++)
    {
        for(uint8_t t = 0; t < AVG_ORDER; t++)
        {
            curr_avg[i] += i_curr_fft_buf[i + t] / (float)AVG_ORDER;
            prev_avg[i] += i_prev_fft_buf[i + t] / (float)AVG_ORDER;
        }
        diff_curr = curr_avg[i] - prev_avg[i];
        if (diff_curr > diff_max) diff_max = diff_curr;
        if (!min_found && curr_avg[i-1] < BW_THRESH && curr_avg[i] > BW_THRESH)
        {
            fmin = i;
            min_found = 1;
        }
    }

    // Find fmax
    for (uint32_t i = conv_samples-1; i >= 1; i--)
    {
        if (curr_avg[i] < BW_THRESH && curr_avg[i-1] > BW_THRESH)
        {
            fmax = i;
            break;
        }
    }

    // Calc Avg, Variance
    for (uint32_t i = fmin+BW_MARGIN; i < fmax-BW_MARGIN; i++)
    {
        gain_avg += curr_avg[i];
    }
    gain_avg /= (float)(fmax-fmin-2*BW_MARGIN);
    for (uint32_t i = fmin+BW_MARGIN; i < fmax-BW_MARGIN; i++)
    {
        var_dif = curr_avg[i] - gain_avg;
        gain_var += var_dif*var_dif;
    }
    gain_var /= (float)(fmax-fmin-2*BW_MARGIN - 1);

    // Set features
    o_features->val[0] = fmax - fmin;
    o_features->val[1] = gain_avg;
    o_features->val[2] = gain_var;
    o_features->val[3] = diff_max;
}
