# sboxgates
[![DOI](https://zenodo.org/badge/79294181.svg)](https://zenodo.org/badge/latestdoi/79294181)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Program for finding low gate count implementations of S-boxes.

The algorithm used is described in [Kwan, Matthew: "Reducing the Gate Count of Bitslice DES."
IACR Cryptology ePrint Archive 2000 (2000): 51](https://ia.cr/2000/051), with improvements from the
GitHub project [SBOXDiscovery](https://github.com/tripcode/SBOXDiscovery) added. In
addition to finding logic circuits using standard (NOT, AND, OR, XOR) gates, the program also
supports ANDNOT gates and 3-bit LUTs. The latter can be used to find efficient implementations for
use on Nvidia GPUs that support the LOP3.LUT instruction, or on FPGAs.

## Dependencies

* MPI
* [msgpack](https://github.com/msgpack/msgpack-c)
* [Graphviz](https://github.com/ellson/graphviz) (for generating visual representations)

## Build

```
sudo apt-get install libmsgpack-dev libopenmpi-dev openmpi-bin
./build.sh
```

## Run

The `-h` command line argument will display a brief list of command line options.

Generate a logic circuit representation of the Rijndael S-box:
```
./sboxgates -b sboxes/rijndael.txt
```

Generate a LUT circuit for output bit 0 of the Rijndael S-box:
```
mpirun ./sboxgates -l -o 0 -b sboxes/rijndael.txt
```

Visualize a generated circuit with Graphwiz:
```
./sboxgates -d 1-067-162-3-c32281db.state | dot -Tpng > 1-067-162-3-c32281db.png
```

Convert a generated circuit to C/CUDA:
```
./sboxgates -c 1-067-162-3-c32281db.state > 1-067-162-3-c32281db.c
```

## License

This project is licensed under the GNU General Public License -- see the [LICENSE](LICENSE)
file for details.
