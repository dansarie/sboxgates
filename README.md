# sboxgates
Program for finding low gate count implementations of S-boxes.

The algorithm used is described in Kwan, Matthew: "Reducing the Gate Count of Bitslice DES."
IACR Cryptology ePrint Archive 2000 (2000): 51. Improvements from
[SBOXDiscovery](https://github.com/DeepLearningJohnDoe/SBOXDiscovery) have been added. In addition
to finding logic circuits using standard (NOT, AND, OR, XOR) gates, the program also supports ANDNOT
gates and 3-bit LUTs. The latter can be used to find efficient implementations for use on Nvidia
GPUs that support the LOP3.LUT instruction, or on FPGAs.

## Dependencies

* [msgpack](https://github.com/msgpack/msgpack-c)

## Build

With MPI (highly recommended when building LUT networks):
```console
$ ./build-mpi.sh
```

Without MPI:
```console
$ ./build.sh
```

## Run

The `-h` command line argument will display a brief list of command line options.

With MPI:
```console
$ mpirun ./sboxgates
```

Without MPI:
```console
$ ./sboxgates
```

## License

This project is licensed under the GNU General Public License - see the [LICENSE](LICENSE)
file for details.
