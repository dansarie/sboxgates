#!/bin/sh

gcc -Ofast -march=native -mtune=native sboxgates.c -Wall -Wpedantic -o sboxgates -fopenmp
