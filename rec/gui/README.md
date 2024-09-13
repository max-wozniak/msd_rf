compile:

g++ -c gui_test.cpp

link:

g++ gui_test.o -o gui_test -lfftw3 -lm -lsfml-graphics -lsfml-window -lsfml-system

run:

./gui_test