#!/bin/sh

mpicc -Ofast -march=native convert_graph.c lut.c sboxgates.c state.c  -Wall -Wpedantic -o sboxgates -lmsgpackc
