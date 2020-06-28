# sticky-HDP-HMM-Vector Auto-Regression

commands to run for c++

g++ -std=c++14 -Wall -Wextra -pedantic -c -fPIC c_extensions.cpp -o c_extensions.o
g++ -shared c_extensions.o -o c_extensions.dylib

