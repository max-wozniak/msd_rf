/*
 * rtl_map, a FFT-based visualizer for RTL-SDR devices. (RTL2832/DVB-T)
 * Copyright (C) 2019-2023 by orhun <https://www.github.com/orhun>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdlib.h>
#include <math.h>
#include <signal.h>
#include <string.h>
#include <complex.h>
#include <unistd.h>
#include <stdarg.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>
/*! External libraries */
#include <fftw3.h>
#include <rtl-sdr.h>
#include "preprocessor.h"
#include "classifier.h"
// ^ also includes classifier.h

#define NUM_READ 512
#define log_info(...) print_log(INFO, __VA_ARGS__)
#define log_error(...) print_log(ERROR, __VA_ARGS__)
#define log_fatal(...) print_log(FATAL, __VA_ARGS__)

static rtlsdr_dev_t *dev; /*!< RTL-SDR devices */
static fftw_plan fftwp; /**!
			 * FFT plan that will contain 
 			 * all the data that FFTW needs 
 			 * to compute the FFT 
 			 */ 
static fftw_complex *in, *out; /*!< Input and output arrays of the transform */
static FILE *file, *rfile; /**!
				  * File to write
                  * File to read
                  * file to read samples
				  */
static FILE *gnuplotPipe; // file pointers for gnuplot pipe
static FILE *mdl;
static struct sigaction sig_act; /*!< For changing the signal actions */
static const int n_read = NUM_READ; /*!< Sample count & data points & FFT size */
static const int buf_size = n_read * 10;
static int n, /*!< Used at raw I/Q data to complex conversion */
	_dev_id = 0,
	read_count = 0, /*!< Current read count */
	out_r, out_i, /*!< Real and imaginary parts of FFT *out values */
	_samp_rate = NUM_READ * 4000, /*!< [ARG] Sample rate (optional) */
	_gain = 14, /*!< [ARG] Device gain (optional) */
	_refresh_rate = 500, /*!< [ARG] Refresh interval for continuous read (optional) */
	_num_read = INT_MAX, /*!< [ARG] Number of reads, set to max val of int (optional) */
	_use_gnuplot = 1, /*!< [ARG] Use gnuplot or not (optional) */
	_cont_read = 0, /*!< [ARG] Continuously read samples from device (optional) */
	_mag_graph = 0, /*!< [ARG] Show magnitude instead of dB (optional) */
	_offset_tuning = 1, /**!
 			     * [ARG] Enable or disable offset tuning for zero-IF tuners
			     * which allows to avoid problems caused by the DC offset 
			     * of the ADCs and 1/f noise. (optional)
			     */
	_log_colors = 1, /*!< [ARG] Use colored flags while logging (optional) */
	_write_file = 0, /*!< [ARG] Write output of the FFT to a file|stdout (optional) */
  _read_file = 0; /*!< [ARG] Read file input time-domain samples instead of device handle (optional) */
static const int y_range = 100; // y-axis range for GNUplot
static int _center_freq;
static const int bandwidth = 12500; // 12.5 kHz
static classifier_t* cls; // structs for classifier
static data_point_t features; // structs for pre-processed features
static uint8_t output; // class model output
static uint8_t output_d1; // class model output d1
static char t_buf[16], /*!< Time buffer, used for getting current time */
	*_filename, /*!< [ARG] File name to write samples (optional) */
	*log_levels[] = { 
		"INFO", "ERROR", "FATAL" /*!< Log levels */
	},
	*level_colors[] = {
		"\x1b[92m", "\x1b[91m", "\x1b[33m" /*!< Log level colors (green, red, yellow) */
	},
	*bold_attr = "\x1b[1m", /*!< Enable bold text in terminal */
	*all_attr_off = "\x1b[0m"; /*!< Clear previous attributes in terminal */
enum log_level {INFO, ERROR, FATAL}; /*!< Log level enumeration */
static va_list vargs;  /*!< Holds information about variable arguments */
static time_t raw_time; /*!< Represents time value */
static char* _in_file = NULL; /*!< Input sample file */

static float sample_bin[NUM_READ];
static float sample_bin_d1[NUM_READ];

// Structure to hold arguments for each thread
typedef struct {
    pthread_t thread_id;
} thread_args_t;

void *async_read_thread(void *arg);

/*!
 * Print log message with time, level and text.
 * Also supports format specifiers.
 *
 * \param level logging level (info:0, error:1, fatal:2)
 * \param format string and format specifiers for vfprintf function  
 * \return 0 on success
 */
static int print_log(int level, char *format, ...){
	raw_time = time(NULL);
	struct tm *local_time = localtime(&raw_time);
	t_buf[strftime(t_buf, sizeof(t_buf), 
			"%H:%M:%S", local_time)] = '\0';
	if (_log_colors)
		fprintf(stderr, "%s[%s] %s%s%s ", 
		bold_attr, 
		t_buf, 
		level_colors[level], 
		log_levels[level], 
		all_attr_off);
	else
		fprintf(stderr, "[%s] %s ", 
		t_buf, 
		log_levels[level]);
  	va_start(vargs, format);
  	vfprintf(stderr, format, vargs);
  	va_end(vargs);
	return 0;
}
/*!
 * Cancel asynchronous read operation on the SDR device. 
 * Close pipe and file.
 * Exit.
 */
static void do_exit() {
	rtlsdr_cancel_async(dev);
	if(_use_gnuplot) {
		pclose(gnuplotPipe);
    }
	if(_filename != NULL && strcmp(_filename, "-"))
		fclose(file);
    destroy_classifier(cls); // free model
	exit(0);
}
/*!
 * Callback for sigaction struct (sig_act).
 * Referenced at register_signals() function.
 *
 * \param signum Incoming signal number from os
 */
static void sig_handler(int signum){
    log_info("Signal caught, exiting...\n");
	do_exit();
}
/*!
 * Set signals and assign them a handler
 * for catching os signals.
 *
 * \return 0 on success
 */
static int register_signals(){
	sig_act.sa_handler = sig_handler;
    sigemptyset(&sig_act.sa_mask);
    sig_act.sa_flags = 0;
    sigaction(SIGINT, &sig_act, NULL);
    sigaction(SIGTERM, &sig_act, NULL);
    sigaction(SIGQUIT, &sig_act, NULL);
	/**!
	* NOTE: Including the SIGPIPE signal might cause 
	* problems with the pipe communication.
	* However, in tests we got any at problems at all.
	*/
    sigaction(SIGPIPE, &sig_act, NULL);
	return 0;
}
/*!
 * Execute gnuplot commands through the opened pipe.
 *
 * \param format string (command) and format specifiers
 * \return 0 on success
 */
static int gnuplot_exec(char *format, ...){
	va_start(vargs, format);
  	vfprintf(gnuplotPipe, format, vargs);
  	va_end(vargs);
	return 0;
}
/*!
 * Open gnuplot pipe.
 * Set labels & title.
 * Set xtics after calculation of X-Axis's starting and ending point.
 * Exits on failure at opening gnuplot pipe.
 * 
 * \return 0 on success
 * \return 1 on given -D argument (don't use gnuplot)
 */
static int configure_gnuplot(void){
	if (!_use_gnuplot)
		return 1;
	gnuplotPipe = popen("gnuplot -persist", "w");
	if (!gnuplotPipe) {
		log_error("Failed to open gnuplot pipe.");
		exit(1);
	}
    gnuplot_exec("set terminal qt\n");
    gnuplot_exec("set title 'Channel %d Spectrum (MHz)' enhanced\n", _dev_id);
    //gnuplot_exec("set xlabel 'Frequency (MHz)'\n");
    gnuplot_exec("set ylabel 'Gain (dB)'\n");
    /**!
    * Compute center frequency in MHz. [Center freq./10^6]
    * Step size = [(512*10^3)/10^6] = 3.854
    * Substract and add step size to center frequency for
    * finding max and min distance from center frequency.
    */
    float center_mhz = _center_freq / pow(10, 6);
    float step_size = (n_read * pow(10, 3))  / pow(10, 6);
    double center_max = center_mhz + (bandwidth / pow(10, 6))/2;
    double center_mid_max = center_mhz + (bandwidth / pow(10, 6))/4;
    double center_mid_min = center_mhz - (bandwidth / pow(10, 6))/4;
    double center_min = center_mhz - (bandwidth / pow(10, 6))/2;
    gnuplot_exec("set xtics ('%.3f' 1, '%.3f' %d, '%.3f' %d, '%.3f' %d, '%.3f' %d)\n", 
        center_min,
        center_mid_min, n_read/4,	
        center_mhz, n_read/2,
	center_mid_max, 3*n_read/4,
        center_max, n_read);
    gnuplot_exec("set yrange [0:%d]\n", y_range);
    gnuplot_exec("set grid\n");
	return 0;
}
/*!
 * Configure RTL-SDR device according to given optarg parameters.
 * Exits on failure at finding device from ID or
 * opening the device with given ID.
 *
 * \return 0 on success
 * \return 1 on buffer reset error
 */
static int configure_rtlsdr(){
    	int device_count = rtlsdr_get_device_count();
	if (!device_count) {
		log_error("No supported devices found.\n");
		exit(1);
	}
	log_info("Starting rtl_map ~\n");
	log_info("Found %d device(s):\n", device_count);
	for(int i = 0; i < device_count; i++) {
		if(_log_colors)
			log_info("#%d: %s%s%s\n", i, bold_attr, rtlsdr_get_device_name(i), all_attr_off);
		else
			log_info("#%d: %s\n", i, rtlsdr_get_device_name(i));
	}
	int dev_open = rtlsdr_open(&dev, _dev_id);
	if (dev_open < 0) {
		log_fatal("Failed to open RTL-SDR device #%d\n", _dev_id);
		exit(1);
	}else{
		log_info("Using device: #%d\n", _dev_id);
	}
	/**!
	 * Set gain mode auto if '_gain' equals to 0.
 	 * Otherwise, set gain mode to manual.
	 * (Mode 1 [manual] needs gain value so 
	 * gain setter function must be called.)
 	 */
	if(!_gain){
		rtlsdr_set_tuner_gain_mode(dev, _gain);
		log_info("Gain mode set to auto.\n");
	}else{
		rtlsdr_set_tuner_gain_mode(dev, 1);
		int gain_count = rtlsdr_get_tuner_gains(dev, NULL);
		log_info("Supported gain values (%d): ", gain_count);
		int gains[gain_count], supported_gains = rtlsdr_get_tuner_gains(dev, gains);
		for (int i = 0; i < supported_gains; i++){
			/**!
			 * Different RTL-SDR devices have different supported gain
			 * values. So select gain value between 1.0 and 3.0
			 */
			if (gains[i] > 10 && gains[i] < 30)
				_gain = gains[i];
			fprintf(stderr, "%.1f ", gains[i] / 10.0);
		}
		fprintf(stderr, "\n");
		log_info("Gain set to %.1f\n", _gain / 10.0);
		rtlsdr_set_tuner_gain(dev, _gain);
	}
	/**! 
	 * Enable or disable offset tuning for zero-IF tuners, which allows to avoid
 	 * problems caused by the DC offset of the ADCs and 1/f noise.
 	 */
	rtlsdr_set_offset_tuning(dev, _offset_tuning);
	rtlsdr_set_center_freq(dev, _center_freq);
	rtlsdr_set_sample_rate(dev, _samp_rate);
    int result = rtlsdr_set_tuner_bandwidth(dev, bandwidth);
    if (result) {
        log_error("RTL-SDR set_tuner_bandwidth failed :(\n");
    } else {
        log_info("Set tuner bandwidth to %d Hz.\n", bandwidth);
    }
	log_info("Center frequency set to %d Hz.\n", _center_freq);
	log_info("Sampling at %d S/s\n", _samp_rate);
	int r = rtlsdr_reset_buffer(dev);
	if (r < 0){
		log_fatal("Failed to reset buffers.\n");
		return 1;
	}
	return 0;
}
/*!
 * Open file with given _filename parameter.
 * Set 'stdout' output if _filename is given as single dash.
 * Exits on failure in opening the file.
 *
 * NOTE: This block is seperated from parse_args() function
 * due to some bugs. (About pipe open synchronization I think)
 * So, we handle open_file after parsing arguments and 
 * opening gnuplot pipe.
 * 
 * \return 0 on success
 */
static int open_file(){
	if (_filename != NULL){
		_write_file = 1;
		if(!strcmp(_filename, "-")) {
        	file = stdout;
		} else {
			file = fopen(_filename, "w+");
			if (!file) {
				log_error("Failed to open %s\n", _filename);
				exit(1);
			}
 		}
	}
  if (_in_file != NULL) {
    _read_file = 1;
    rfile = fopen(_in_file, "r+");
    if (!rfile) {
      log_error("Failed to open %s\n", _in_file);
      exit(1);
    }
  }
	return 0;
}

/*!
 * Create FFT graph from raw I/Q samples read from RTL-SDR.
 * Uses gnuplot for creating graph. (optional, see -D arg.)
 * Uses fftw3 library for FFT's computations.
 *
 * \param sample_c sample count, also used for FFT size
 * \param buf array that contains I/Q samples
 */
static void create_fft(int sample_c, uint8_t *buf){
    if (_use_gnuplot) {
        //gnuplot_exec("set terminal qt %d\n", _dev_id);
        //gnuplot_exec("set title 'Channel %d Spectrum (MHz)' enhanced\n", _dev_id + 1);
        ////gnuplot_exec("set xlabel 'Frequency (MHz)'\n");
        //gnuplot_exec("set ylabel 'Gain (dB)'\n");
        ///**!
        //* Compute center frequency in MHz. [Center freq./10^6]
        //* Step size = center_mhz - start
        //* Substract and add step size to center frequency for
        //* finding max and min distance from center frequency.
        //*/
        //double center_mhz = _center_freq_arr[_dev_id] / pow(10, 6);
        //double start = center_mhz - (bandwidth / pow(10, 6)) / 2;
        //double end = center_mhz + (bandwidth / pow(10, 6)) / 2;
        //double step_size = center_mhz - start;
        //gnuplot_exec("set xtics ('%.4f' 1, '%.4f' %d, '%.1f' %d, '%.4f' %d, '%.4f' %d)\n", 
        //    start,  	       // min
	    //center_mhz-step_size/2, n_read/4,  // min-middle
        //    center_mhz, n_read/2, 	       // middle
	    //center_mhz+step_size/2, 3*n_read/4,// middle-max
        //    end, n_read);     // max
        //gnuplot_exec("set yrange [0:%d]\n", y_range);
        //gnuplot_exec("set grid\n");
    }
    float amp, db; /*!< Amplitude & dB */
	/**! 
	 * Configure FFTW to convert the samples in time domain to frequency domain. 
	 * Allocate memory for 'in' and 'out' arrays.
	 * fftw_complex type is a basically double[2] that composed of the 
	 * real (in[i][0]) and imaginary (in[i][1]) parts of a complex number.
	 * in -> Complex numbers processed from 8-bit I/Q values.
	 * out -> Output of FFT (computed from complex input).
	 */
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*sample_c);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*sample_c);
	/**!
	 * Declare FFTW plan which is responsible for having in and out data.
	 * First parameter (sample_c) -> FFT size 
	 * FFTW_FORWARD/FFTW_BACKWARD -> Indicates the direction of the transform.
	 * Technically, sign of the exponent in the transform.
	 * FFTW_MEASURE/FFTW_ESTIMATE
	 * Use FFTW_MEASURE if you want to execute several FFTs and find the 
	 * best computation in certain amount of time. (Usually a few seconds)
	 * FFTW_ESTIMATE is the contrary. Does not run any computation, just
	 * builds a reasonable plan.
	 * If you are dealing with many transforms of the same FFT size and
	 * initialization time is not important, use FFT_MEASURE. 
	 */
    //printf("hi :3. thread = %d\n", _dev_id);
    fftwp = fftw_plan_dft_1d(n_read, in, out, FFTW_FORWARD, FFTW_MEASURE);
	/**!
	 * Convert buffer from IQ to complex ready for FFTW.
	 * RTL-SDR outputs 'IQIQIQ...' so we have to read two samples 
	 * at the same time. 'n' is declared for this approach.
	 * Sample is 127 for zero signal, so substract ~127.34 for exact value.
	 * Loop through samples and fill 'in' array with complex samples.
	 * 
	 * NOTE: There is a common issue with cheap RTL-SDR receivers which
	 * is 'center frequency spike' / 'central peak' problem related to 
	 * I/Q imbalance. This problem can be solved with a implementation of 
	 * some algorithms.
	 * More detail: 
	 * https://github.com/roger-/pyrtlsdr/issues/94
	 * https://wiki.analog.com/resources/eval/user-guides/ad-fmcomms1-ebz/iq_correction
	 *
	 * TODO #1: Implement I/Q correction
	 */
    //for (int i = 0; i < n_read; i++) {\
    //    printf("Frame %d, i = %d, raw data = %d \n", read_count[_dev_id], i, buf[i]);
    //}
	n = 0;
	for (int i=0; i<sample_c; i+=2){
        //printf("%f + %f * i\n", (buf[n[_dev_id]]-127.34), (buf[n[_dev_id]+1]-127.34));
		in[i] = (buf[n]-127.34) + (buf[n+1]-127.34) * I;
        //printf("Frame %d, i = %d, n = %d, fft in data = %f + %f * i \n", read_count[_dev_id], i, n[_dev_id], creal(in[_dev_id][i]), cimag(in[_dev_id][i]));
		n++;
	}
	/**! 
	 * Convert the complex samples to complex frequency domain.
	 * Compute FFT.
	 */
    fftw_execute(fftwp); 
	if(!_cont_read && _use_gnuplot)
		log_info("Creating FFT graph from samples using gnuplot...\n");
	else if (!_cont_read && !_use_gnuplot)
		log_info("Reading samples...\n");
	if(_use_gnuplot) {
        // Replot without resetting anything
        gnuplot_exec("plot '-' smooth frequency with linespoints lt -1 notitle\n");
    }
	for (int i=0; i < n_read; i++){
		out_r = creal(out[i]) * creal(out[i]);
		out_i = cimag(out[i]) * cimag(out[i]);
		amp = sqrt(out_r + out_i);
		if (!_mag_graph)
			db = 20 * log10(amp);
		else
			db = amp;
        //printf("Frame %d, i = %d, data = %d \n", read_count, i, db);
		if(_write_file)
			fprintf(file, "%d	%f\n", i+1, db);
		if(_use_gnuplot)
			gnuplot_exec("%d   %f\n", db, i+1);
		/**! 
		 * Fill sample_bin with values.
		 *
		 * TODO #2: Find the maximum value of samples, show it on graph 
		 * with a different color.  Might be useful for frequency scanner.
		 * If you want to sort values see qsort function.
		 * Example code: qsort(sample_bin, n_read, sizeof(Bin), cmp_sample);
		 */
		// fill previous spectrum
		sample_bin_d1[i] = sample_bin[i];
		// fill current spectrum
		sample_bin[i] = (isfinite(db)) ? db : 0;
	}
	if(_use_gnuplot){
		/**!
		 * Stop giving points to gnuplot with 'e' command.
		 * Have to flush the output buffer for [read -> graph] persistence.
		 */
		gnuplot_exec("e\n");
		fflush(gnuplotPipe);
        
        // preprocess the data
        extract_features(&features, sample_bin, sample_bin_d1, n_read);
        //for (int i = 0; i < n_read; i = i + 1) {
        //    printf("sample: %f, sample_d1: %f\n", sample_bin[_dev_id][i], sample_bin_d1[_dev_id][i]);
        //}
        for (int i = 0; i < NUM_FEATURES; i = i + 1) {
            printf("%.5f ", features.val[i]);
        }
        //printf("\n");
        // send the features into the model
        output = classify(cls, &features);
        printf("output: %d output_d1: %d\n", output, output_d1);
        
  		//  GET CLASSIFICATION RESULTS, SEND TO CURRENT WINDOW
        // only update window if there was a change in class. result
	if (output != output_d1) {
            gnuplot_exec("unset label \n");
            if (output) {
                gnuplot_exec("set label 'Channel is transmitting!' at graph 0.5,0.95 tc \"red\" font \"sans,15\"\n");
            } else {
                gnuplot_exec("set label 'Channel is not transmitting!' at graph 0.5,0.95 tc \"red\" font \"sans,15\"\n");
            }
	}
        // propagate old class result
        output_d1 = output;
	}
	/**!
	 * Deallocate FFT plan.
	 * Free 'in' and 'out' memory regions.
	 */
    fftw_free(in); 
    fftw_free(out);
    fftw_destroy_plan(fftwp);
	read_count++;
}
/*!
 * Asynchronous read callback.
 * Program jump to this function after read operation finished.
 * Runs create_fft() function and provides continuous read
 * depending on the -C argument with refresh rate.
 * Exits if -C argument is not given.
 *
 * \param n_buf raw I/Q samples
 * \param len length of buffer
 * \param ctx context which is given at rtlsdr_read_async(...)
 */
static void async_read_callback(uint8_t *n_buf, uint32_t len, void *ctx) {
    // Cast the context to the thread_ctx_t structure to retrieve the thread ID
    //for (int i = 0; i < n_read; i++) {\
    //    printf("Frame %d, i = %d, data = %d \n", read_count[thread_id], i, n_buf[i]);
    //}
	create_fft(n_read, n_buf);
	if (_cont_read && read_count < _num_read){
        //current_dev = !current_dev; // ping-pong between devices to read from
        usleep(1000*_refresh_rate);
        rtlsdr_read_async(dev, async_read_callback, NULL, 0, buf_size);
	}else{
		log_info("Done, exiting...\n");
		do_exit();
	}
}

/*!
 * 1. Read set of samples from file
 * 2. if read is successful, send to create_fft. otherwise, break
 * 3. after create_fft, sleep for the provided refresh rate
 * 4. repeat
 */
static void read_plot_data() {
	uint8_t file_buf [n_read];
	size_t samples = fread(file_buf, sizeof(uint8_t), n_read, rfile);
	while (samples == n_read) {
		create_fft(n_read, file_buf);
		usleep(1000*_refresh_rate);
		samples = fread(file_buf, sizeof(uint8_t), n_read, rfile);
	}
	log_info("Done, exiting...\n");
	do_exit();
}

/*!
 * Print usage and exit.
 */
static void print_usage() {
	char *usage	= "rtl_map, a FFT-based visualizer for RTL-SDR devices. (RTL2832/DVB-T)\n\n"
                  "Usage:\t[-d device index (default: 0)]\n"
                  "\t[-s sample rate (default: 2048000 Hz)]\n"
		  "\t[-f center frequency (Hz)] *\n"
				  "\t[-g gain (0 for auto) (default: ~1-3)]\n"
				  "\t[-n number of reads (default: int_max.)]\n"
				  "\t[-r refresh rate for -C read (default: 500ms)]\n"
				  "\t[-D don't show gnuplot graph (default: show)]\n"
				  "\t[-C continuously read samples (default: off)]\n"
				  "\t[-M show magnitude graph (default graph: dB)]\n"
				  "\t[-O disable offset tuning (default: on)]\n"
				  "\t[-T turn off log colors (default: on)]\n"
				  "\t[-h show this help message and exit]\n"
				  "\t[-i input filename for reading TD samples from a bin file]\n"
                  "\t[filename (a '-' dumps samples to stdout)]\n\n";
    fprintf(stderr, "%s", usage);
    exit(0);
}
/*!
 * Parse command line arguments.
 *
 * \param argc argument count
 * \param argv argument vector
 * \return 0 on success
 */
static int parse_args(int argc, char **argv) {
	int opt;
	while ((opt = getopt(argc, argv, "d:s:f:g:r:n:DCMOThi:")) != -1) {
        switch (opt) {
            case 'd':
                _dev_id = atoi(optarg);
                break;
			case 's':
                _samp_rate = atoi(optarg);
            	break;
			case 'f':
		_center_freq = atoi(optarg);
		break;
            	case 'g':
				/**! Tenths of a dB */
                _gain = (int)(atof(optarg) * 10); 
                break;
			case 'r':
                _refresh_rate = atoi(optarg);
                break;
			case 'n':
                _num_read = atoi(optarg);
                break;
			case 'D':
                _use_gnuplot = 0;
                break;		
			case 'C':
                _cont_read = 1;
                break;
			case 'M':
                _mag_graph = 1;
                break;
			case 'O':
                _offset_tuning = 0;
                break;
			case 'T':
                _log_colors = 0;
                break;
            case 'h':
                print_usage();
                break;
            case 'i':
				_in_file = (char *) malloc(strlen(optarg) + 1);
				if (_in_file == NULL) {
					fprintf(stderr, "Memory allocation failed\n");
					exit(1);
				}
				strcpy(_in_file, optarg);
				_cont_read = 1;
				break;
            default:
                print_usage();
                break;
        }
    }
	if (!_center_freq) {
		print_usage();
	}
	_filename = argv[optind];
	return 0;
}


/*!
 * Entry point (main)
 * Self-explanatory.
 *
 * \param argc argument count
 * \param argv argument vector
 */
void main(int argc, char **argv){
	parse_args(argc, argv);
    
    // initialize classification models
    mdl = fopen("../../../classifier/model.mdl", "r");
    cls = init_classifier(mdl);
    fclose(mdl);
    if(cls == NULL) {
        log_error("Failed to open classification model.");
        exit(1);
    }
    output_d1 = 2;
    
    
	if (_in_file == NULL) {
		register_signals();
		configure_gnuplot();
		configure_rtlsdr();
        open_file();
        
	int val = rtlsdr_read_async(dev, async_read_callback, NULL, 0, buf_size);
	log_info("Return val: %d\n", val);

	} else {
		register_signals();
		//configure_gnuplot(0);
		//configure_rtlsdr();
		open_file();
		read_plot_data();
		fclose(rfile);
		free(_in_file);
	}
    
}
