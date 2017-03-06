# sboxgates
Program for finding low gate count implementations of S-boxes.

The algorithm used is described in Kwan, Matthew: "Reducing the Gate Count of Bitslice DES."
IACR Cryptology ePrint Archive 2000 (2000): 51. Currently, the program supports finding gate
networks using standard gates (NOT, AND, OR, XOR) and 3-bit LUTs.

## Build

With OpenMP (highly recommended):
```console
$ gcc -Ofast -march=native -fopenmp sboxgates.c -o sboxgates
```

Without OpenMP:
```console
$ gcc -Ofast -march=native sboxgates.c -o sboxgates
```
