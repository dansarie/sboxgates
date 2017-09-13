# sboxgates
Program for finding low gate count implementations of S-boxes.

The algorithm used is described in [Kwan, Matthew: "Reducing the Gate Count of Bitslice DES."
IACR Cryptology ePrint Archive 2000 (2000): 51](ia.cr/2000/051), with improvements from the GitHub
project [SBOXDiscovery](https://github.com/DeepLearningJohnDoe/SBOXDiscovery) added. In addition to
finding logic circuits using standard (NOT, AND, OR, XOR) gates, the program also supports ANDNOT
gates and 3-bit LUTs. The latter can be used to find efficient implementations for use on Nvidia
GPUs that support the LOP3.LUT instruction, or on FPGAs.

## Dependencies

* [msgpack](https://github.com/msgpack/msgpack-c)
* [Graphviz](https://github.com/ellson/graphviz) (for generating visual representations)

## Build

Edit [sboxgates.c](sboxgates.c) and change the constant array `g_target_sbox` to the lookup table
for the S-box you wish to generate logic circuits for.

Compile with MPI (highly recommended when building LUT networks):
```console
$ ./build-mpi.sh
```

Compile without MPI:
```console
$ ./build.sh
```

## Run

The `-h` command line argument will display a brief list of command line options.

Generate a logic circuit representation or the S-box:
```console
$ ./sboxgates
```

Generate a LUT circuit for output bit 0 of the S-box (using MPI):
```console
$ mpirun ./sboxgates -l -o 0
```

Visualize a generated circuit with Graphwiz:
```console
$ ./sboxgates -d 1-067-162-3-c32281db.state | dot -Tpng > 1-067-162-3-c32281db.png
```

Convert a generated circuit to C/CUDA:
```console
$ ./sboxgates -c 1-067-162-3-c32281db.state > 1-067-162-3-c32281db.c
```

## License

This project is licensed under the GNU General Public License - see the [LICENSE](LICENSE)
file for details.
