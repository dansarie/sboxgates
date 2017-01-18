#!/bin/sh

gcc -Ofast -march=native -mtune=native sboxgates.c -lpthread -Wall -Wpedantic -o sboxgates
