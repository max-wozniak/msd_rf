#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// WINDOW SIZE
#define N 20000

#define M_PI    3.1415926535897932384626433
#define EPSILON 0.00000005f

int main (void) {
	fftw_complex *in, *out, *in2;
	fftw_plan p, q;
	int i;
	float tim;
	clock_t start, end;
	
	// N = 20000
	in  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
	in2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
	// when used in our application, use the FFTW_MEASURE flag instead for optimal performance
	p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	q = fftw_plan_dft_1d(N, out, in2, FFTW_BACKWARD, FFTW_ESTIMATE);
	// populate in with cosine wave
	for (i = 0; i < N; i++) {
		// REAL COMPONENT
		in[i][0] = cos(433050000*2*M_PI*i/N) + 5*cos(433150000*2*M_PI*i/N+M_PI/2.0); 
		// IMAGINARY COMPONENT
		in[i][1] = 0;
	}
	// execute FFT!
	start = clock();
	fftw_execute(p);
	end = clock();
	tim = (float) (end - start) * 1000 / (float) CLOCKS_PER_SEC;
	// execute IFFT!
	fftw_execute(q);

	// print original signal
	printf("ORIGINAL INPUT:\n");
	for (i = 0; i < N; i++)
		printf("index: %d, val = %+9.5f %+9.5f I\n", i, in[i][0], in[i][1]);
	// print freq. domain signal
	printf("FFT OUTPUT:\n");
	for (i = 0; i < N; i++)
		printf("index: %d, val = %+9.5f %+9.5f I\n", i, out[i][0], out[i][1]);

	// normalize recovered signal
	for (i = 0; i < N; i++) {
		in2[i][0] *= 1./N;
		in2[i][1] *= 1./N;
	}

	// print recovered signal
	printf("IFFT OUTPUT:\n");
	for (i = 0; i < N; i++)
		printf("index: %d, val = %+9.5f %+9.5f I\n", i, in2[i][0], in2[i][1]);
		
	// self checking stuff
	printf("\n");
	for (i = 0; i < N; i++) {
		if ((fabs(in[i][0] - in2[i][0]) >= EPSILON) || (fabs(in[i][1] - in2[i][1]) >= EPSILON)) {
			printf("ERROR! Input signal does not match reconstructed!\n");
			return 1;			
		} 	
	}	
	printf("PASS! Input signal matches reconstructed!\n");
	printf("Total time to compute FFT: %+9.5f ms\n", tim);

	fftw_destroy_plan(p);
	fftw_destroy_plan(q);
	fftw_free(in);
	fftw_free(out);
	fftw_free(in2);

	return 0;
}
