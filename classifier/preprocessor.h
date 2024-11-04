#include <stdint.h>
#include "classifier.h"

void extract_features(data_point_t* o_features,float* i_curr_fft_buf, float* i_prev_fft_buf, uint32_t i_num_samples);