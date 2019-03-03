#!/bin/sh

gcc -Ofast -march=native sboxgates.c state.c convert_graph.c -Wall -Wpedantic -o sboxgates -lmsgpackc
