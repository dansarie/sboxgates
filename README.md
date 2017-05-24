# sboxgates
Program for finding low gate count implementations of S-boxes.

The algorithm used is described in Kwan, Matthew: "Reducing the Gate Count of Bitslice DES."
IACR Cryptology ePrint Archive 2000 (2000): 51. In addition to finding networks using standard
(NOT, AND, OR, XOR) gates, the program also supports ANDNOT gates and 3-bit LUTs. The latter can be
used to find efficient implementations for use on Nvidia GPUs that support the LOP3.LUT instruction,
or on FPGAs.

## Build

```console
$ gcc -Ofast -march=native sboxgates.c state.c -o sboxgates -lmsgpackc
```
