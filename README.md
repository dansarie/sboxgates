# sboxgates
Program for finding low gate count implementations of S-boxes.

The algorithm used is described in Kwan, Matthew: "Reducing the Gate Count of Bitslice DES."
IACR Cryptology ePrint Archive 2000 (2000): 51. In addition to finding networks using standard
(NOT, AND, OR, XOR) gates, the program also supports ANDNOT gates and 3-bit LUTs. The latter can be
used to find efficient implementations for use on Nvidia GPUs that support the LOP3.LUT instruction,
or on FPGAs.

## Build

With OpenMP (highly recommended):
```console
$ gcc -Ofast -march=native -fopenmp sboxgates.c -o sboxgates -lmsgpackc
```

Without OpenMP:
```console
$ gcc -Ofast -march=native sboxgates.c -o sboxgates -lmsgpackc
```
