#!/bin/sh

gcc -Ofast -march=native sboxgates.c state.c -Wall -Wpedantic -o sboxgates -lmsgpackc

