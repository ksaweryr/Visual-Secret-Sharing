all: main.o imps.o
	g++ imps.o -o vss main.o -lOpenCL

main.o: main.cpp
	g++ -c -Wall -Wextra -Wpedantic -O3 main.cpp

imps.o: imps.cpp
	g++ -c -Wall -Wextra -Wpedantic -O3 imps.cpp