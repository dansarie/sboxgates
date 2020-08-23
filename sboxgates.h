/* sboxgates.h

   Copyright (c) 2019-2020 Marcus Dansarie

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program. If not, see <http://www.gnu.org/licenses/>. */

#ifndef __SBOXGATES_H__
#define __SBOXGATES_H__

#include <mpi.h>
#include "boolfunc.h"
#include "state.h"

#define MAX_NAME_LEN (1000)

/* Holds all options set by the user. */
typedef struct {
   char fname[MAX_NAME_LEN];     /* Graph file name for generating C or DOT output. */
   char gfname[MAX_NAME_LEN];    /* Partial graph file name. */
   char sboxfname[MAX_NAME_LEN]; /* S-box file name */
   int iterations;               /* Number of iterations per step. */
   int oneoutput;                /* Set to 0-8 if only one output should be generated, else -1. */
   int permute;                  /* Set to 1-255 if S-box should be XOR permuted. */
   metric metric;                /* The graph metric to use. */
   bool output_c;                /* Set to true to convert graph to C function. */
   bool output_dot;              /* Set to true to convert graph to DOT graph. */
   bool lut_graph;               /* Set to true to build 3LUT graph. */
   bool randomize;               /* Set to true to use randomization at various steps. */
   boolfunc avail_gates[17];     /* Available two-input gates. */
   boolfunc avail_not[49];       /* Available two-input gates with inverted input/output. */
   boolfunc avail_3[256];        /* Available three-input gates. */
   int num_avail_3;              /* Number of available three-input gates. */
   int verbosity;                /* How much information should be printed to the terminal. */
} options;

/* Used to broadcast work to be done by other MPI ranks. */
typedef struct {
  state st;
  ttable target;
  ttable mask;
  int8_t inbits[8];
  bool quit;        /* Set to true to signal workers to quit. */
} mpi_work;

MPI_Datatype g_mpi_work_type; /* MPI type for mpi_work struct. */

/* Adds a three input LUT with function func to the state st. Returns the gate number of the
   added LUT. */
gatenum add_lut(state *st, uint8_t func, ttable table, gatenum gid1, gatenum gid2, gatenum gid3);

/* Used to check if any solutions with smaller metric are possible. Uses either the add or the
   add_sat parameter depending on the current metric in use. Returns true if a solution with the
   provided metric is possible with respect to the value of st->max_gates or st->max_sat_metric. */
bool check_num_gates_possible(state *st, int add, int add_sat, const options *opt);

/* Returns true if the truth table is all-zero. */
bool ttable_zero(ttable tt);

/* Performs a masked test for equality. Only bits set to 1 in the mask will be tested. */
bool ttable_equals_mask(const ttable in1, const ttable in2, const ttable mask);

/* Returns the number of input gates in the state. */
int get_num_inputs(const state *st);

/* Generates pseudorandom 64 bit strings. Used for randomizing the search process. */
uint64_t xorshift1024();

/* If sbox is true, a target truth table for the given bit of the sbox is generated.
   If sbox is false, the truth table of the given input bit is generated. */
ttable generate_target(uint8_t bit, bool sbox);

#endif /* __SBOXGATES_H__ */
