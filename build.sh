#!/bin/sh

gcc -Ofast -march=native sboxgates.c -Wall -Wpedantic -o sboxgates -lmsgpackc
