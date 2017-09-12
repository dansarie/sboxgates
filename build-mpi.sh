#!/bin/sh

mpicc -Ofast -march=native sboxgates.c state.c -Wall -Wpedantic -o sboxgates -lmsgpackc -DUSE_MPI
