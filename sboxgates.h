/* sboxgates.h

   Copyright (c) 2019-2021 Marcus Dansarie

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

#include <inttypes.h>
#ifndef NO_MPI_HEADER
#include <mpi.h>
#endif /* NO_MPI_HEADER */
#include <stdlib.h>
#include "boolfunc.h"
#include "state.h"

#define MAX_NAME_LEN (1000)

#define ASSERT_AND_RETURN(R, T, S, M) \
  { \
    gatenum aar_ret = R; \
    ttable aar_target = T; \
    state *aar_st = S; \
    ttable aar_mask = M; \
    if (aar_ret == NO_GATE || ttable_equals_mask(aar_target, (aar_st)->gates[aar_ret].table, \
         aar_mask)) { \
      return aar_ret; \
    } else { \
      fprintf(stderr, "Return assertion in %s failed: %s:%d.\n", __func__, __FILE__, __LINE__); \
      abort(); \
    } \
  }

extern uint8_t g_sbox_enc[256];  /* Target S-box. */

/* Holds all options set by the user. */
typedef struct {
   char fname[MAX_NAME_LEN];     /* Input file name. */
   char gfname[MAX_NAME_LEN];    /* Partial graph file name. */
   int iterations;               /* Number of iterations per step. */
   int oneoutput;                /* Set to 0-8 if only one output should be generated, else -1. */
   int permute;                  /* Set to 1-255 if S-box should be XOR permuted. */
   metric metric;                /* The graph metric to use. */
   bool output_c;                /* Set to true to convert graph to C function. */
   bool output_dot;              /* Set to true to convert graph to DOT graph. */
   bool lut_graph;               /* Set to true to build 3LUT graph. */
   bool randomize;               /* Set to true to use randomization at various steps. */
   bool try_nots;                /* Set to true to generate functions by appending NOT gates. */
   boolfunc avail_gates[17];     /* Available two-input gates. */
   boolfunc avail_not[49];       /* Available two-input gates with inverted input/output. */
   boolfunc avail_3[256];        /* Available three-input gates. */
   int num_avail_3;              /* Number of available three-input gates. */
   int verbosity;                /* How much information should be printed to the terminal. */
} options;

/* Used to broadcast work to be done by other MPI ranks. */
typedef struct {
  state st;         /* The current search state. */
  ttable target;    /* The search target truth table. */
  ttable mask;      /* The current search mask. */
  int8_t inbits[8]; /* List of input bits already used for multiplexing. Terminated by -1. */
  bool quit;        /* Set to true to signal workers to quit. */
  int verbosity;    /* Current verbosity level. */
} mpi_work;

#ifndef NO_MPI_HEADER
extern MPI_Datatype g_mpi_work_type; /* MPI type for mpi_work struct. */
#endif /* NO_MPI_HEADER */

/* Adds a three input LUT gate to the state st. Returns the gate number of the added LUT, or
   NO_GATE.
   st    - pointer to the state struct where the LUT should be added.
   func  - the function, i.e. lookup table, of the added LUT gate.
   table - truth table of the added LUT.
   gid1  - gate number of input 1.
   gid2  - gate number of input 2.
   gid3  - gate number of input 3. */
gatenum add_lut(state *st, uint8_t func, ttable table, gatenum gid1, gatenum gid2, gatenum gid3);

/* Used to check if any solutions with smaller metric are possible. Uses either the add or the
   add_sat parameter depending on the current metric in use. Returns true if a solution with the
   provided metric is possible with respect to the value of st->max_gates or st->max_sat_metric.
   st      - pointer to the search state to check.
   add     - the number of added gates to check for.
   add_sat - the added SAT metric to check for.
   opt     - pointer to options struct. */
bool check_num_gates_possible(const state *st, int add, int add_sat, const options *opt);

/* Returns true if the truth table is all-zero.
   tt - a truth table. */
bool ttable_zero(const ttable tt);

/* Performs a masked test for equality. Only bits set to 1 in the mask will be tested.
   in1  - a truth table.
   in2  - a truth table.
   mask - a mask. */
bool ttable_equals_mask(const ttable in1, const ttable in2, const ttable mask);

/* Returns a pseudorandom 64 bit string. Uses the xorshift1024 algorithm, initialized by
   /dev/urandom. Used in various places to randomize the search process. */
uint64_t xorshift1024();

#endif /* __SBOXGATES_H__ */
