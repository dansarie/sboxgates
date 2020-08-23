# sboxgates
[![DOI](https://zenodo.org/badge/79294181.svg)](https://zenodo.org/badge/latestdoi/79294181)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Build Status](https://travis-ci.com/dansarie/sboxgates.svg?branch=master)](https://travis-ci.com/dansarie/sboxgates)

Program for finding low gate count implementations of S-boxes.

The algorithm used is described in [Kwan, Matthew: "Reducing the Gate Count of Bitslice DES."
IACR Cryptology ePrint Archive 2000 (2000): 51](https://ia.cr/2000/051), with improvements from the
GitHub project [SBOXDiscovery](https://github.com/tripcode/SBOXDiscovery) added. The program
supports searching for gates using any subset of the 16 standard two-input boolean gates.
Additionally, the program also supports 3-bit LUTs. The latter can be used to find efficient
implementations for use on Nvidia GPUs that support the LOP3.LUT instruction, or on FPGAs.

## Dependencies

* [CMake](https://github.com/Kitware/CMake) (for build)
* [libxml2](https://github.com/GNOME/libxml2)
* MPI
* [Graphviz](https://github.com/ellson/graphviz) (for generating visual representations)

## Build

```
sudo apt-get install cmake graphviz libmpich-dev libxml2-dev mpich
mkdir build
cd build
cmake ..
make
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

Visualize a generated circuit with Graphviz:
```
./sboxgates -d 1-067-162-3-c32281db.xml | dot -Tpng > 1-067-162-3-c32281db.png
```

Convert a generated circuit to C/CUDA:
```
./sboxgates -c 1-067-162-3-c32281db.xml > 1-067-162-3-c32281db.c
```

### Selecting gates

The `-a` command line argument is used to specify the two-input gates gates that are available for
the search. The argument value is a bitfield, where each bit represents one gate type. To specify
the gates to be used, add up their values from the table below and pass the sum as the value of
the `-a` argument. If no `-a` argument is specified, the default is 194, i.e. AND, OR, and XOR.

| Gate        | Value |
| ----------- | ----- |
| FALSE       |     1 |
| AND         |     2 |
| A AND NOT B |     4 |
| A           |     8 |
| NOT A AND B |    16 |
| B           |    32 |
| XOR         |    64 |
| OR          |   128 |
| NOR         |   256 |
| XNOR        |   512 |
| NOT B       |  1024 |
| A OR NOT B  |  2048 |
| NOT A       |  4096 |
| NOT A OR B  |  8192 |
| NAND        | 16384 |
| TRUE        | 32768 |

## License

This project is licensed under the GNU General Public License -- see the [LICENSE](LICENSE)
file for details.
