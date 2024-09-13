#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <SFML/Graphics.hpp>

// WINDOW SIZE
#define N 150

//#define M_PI    3.1415926535897932384626433
#define EPSILON 0.00000005f

int main (void) {
	fftw_complex *in, *out;
	fftw_plan p;
	
	// N = 20000
	in  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
	// when used in our application, use the FFTW_MEASURE flag instead for optimal performance
	p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	// populate in with cosine wave
	for (int i = 0; i < N; i++) {
		// REAL COMPONENT
		in[i][0] = cos(50*2*M_PI*i/N) + 5*cos(2*2*M_PI*i/N+M_PI/2.0); 
		// IMAGINARY COMPONENT
		in[i][1] = 0;
	}
	// execute FFT!
	fftw_execute(p);
	
	int windowWidth = 800;
	int windowHeight = 600;

	// create window
	sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight), "Frequency Spectrum");
	sf::VertexArray graph(sf::LineStrip);
	
	// Plot the frequency spectrum
	for (int i = 0; i < N; i++) {
		// Calculate y value for the current x
		float y = (float) out[i][0];
		
		// Map x and y values to window coordinates
		float xPos = ((float) i) / N * windowWidth;
		float yPos = (windowHeight - (y + 1) / 2 * windowHeight) + (windowHeight * 0.4);
		
		// Add point to graph
		graph.append(sf::Vertex(sf::Vector2f(xPos, yPos), sf::Color::Red));
	}

	// Main loop
	while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }

        window.clear(sf::Color::White);
        window.draw(graph);
        window.display();
	}

	fftw_destroy_plan(p);
	fftw_free(in);
	fftw_free(out);

	return EXIT_SUCCESS;
}
