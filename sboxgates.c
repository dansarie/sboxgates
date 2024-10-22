/* sboxgates.c

   Program for finding low gate count implementations of S-boxes.
   The algorithm used is described in Kwan, Matthew: "Reducing the Gate Count of Bitslice DES."
   IACR Cryptology ePrint Archive 2000 (2000): 51. Improvements from
   SBOXDiscovery (https://github.com/DeepLearningJohnDoe/SBOXDiscovery) have been added.

   Copyright (c) 2016-2017, 2019-2021 Marcus Dansarie

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

#include <argp.h>
#include <assert.h>
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include "convert_graph.h"
#include "lut.h"
#include "sboxgates.h"
#include "state.h"

uint8_t g_sbox_enc[256] = {0};        /* Defined in sboxgates.h. */

ttable g_target[8];           /* Truth tables for the output bits of the sbox. */
MPI_Datatype g_mpi_work_type; /* MPI type for mpi_work struct. Defined in sboxgates.h. */

const char *argp_program_version = "sboxgates 1.0";
const char *argp_program_bug_address = "https://github.com/dansarie/sboxgates/issues";
const char doc[] = "Generates graphs of Boolean gates or 3-input LUTs that realize a specified "
    "S-box. Generated graphs can be converted to C/CUDA source code or to Graphviz DOT format.\v"
    "This program uses MPI for parallelization and should therefore be run using the mpirun "
    "utility. Generated graphs are output as XML files. In its basic mode, the program generates a "
    "single graph for all outputs of the S-box. It is also possible to generate separate graphs "
    "for each output, which can significantly decrease the time to generate the graph. ";
const char args_doc[] = "INPUT_FILE";
struct argp_option argp_options[] = {
  {0,                1000,            0, 0, "Graph generation", 1},
  {"available-gates", 'a', "gates",      0, "Specify the set of available gates "
                                            "(bitfield 0-65535).", 1},
  {"graph",           'g', "graph",      0, "Load graph from file as initial state. "
                                            "(For use with -o.)", 1},
  {"iterations",      'i', "iterations", 0, "Set number of iterations per step.", 1},
  {"lut",             'l',            0, 0, "Generate LUT graph. Results in smaller graphs but "
                                            "takes significantly more time.", 1},
  {"append-not",      'n',            0, 0, "Try to generate more boolean functions by appending "
                                            "NOT gates.", 1},
  {"single-output",   'o', "output",     0, "Generate single-output graph for specified output.",
      1},
  {"permute",         'p', "value",      0, "Permute the input S-box by XORing it with value.", 1},
  {"sat-metric",      's',            0, 0, "Use graph size metric which attempts to optimize the "
                                            "generated graph for use with SAT solvers.", 1},
  {"verbose",         'v',            0, 0, "Increase verbosity.", 1},
  {0,                1001,            0, 0, "Graph conversion", 2},
  {"convert-c",       'c',            0, 0, "Convert input file to a C or CUDA function.", 2},
  {"convert-dot",     'd',            0, 0, "Convert input file to a DOT digraph.", 2},
  {0}
};

/* Returns true if the truth table is all-zero. */
bool ttable_zero(ttable tt) {
  for(size_t i = 0; i < sizeof(ttable) / sizeof(uint64_t); i++) {
    if(tt[i]) {
      return false;
    }
  }
  return true;
}

/* Test two truth tables for equality. */
static inline bool ttable_equals(const ttable in1, const ttable in2) {
  return ttable_zero(in1 ^ in2);
}

/* Performs a masked test for equality. Only bits set to 1 in the mask will be tested. */
bool ttable_equals_mask(const ttable in1, const ttable in2, const ttable mask) {
  return ttable_zero((in1 ^ in2) & mask);
}

/* Adds a gate to the state st. Returns the gate id of the added gate. If an input gate is
   equal to NO_GATE (only gid1 in case of a NOT gate), NO_GATE will be returned. */
static gatenum add_gate(state * restrict st, gate_type type, gatenum gid1, gatenum gid2,
    const options * restrict opt) {
  assert(!(type == NOT && gid2 != NO_GATE));
  assert(type != IN && type != LUT);
  assert(gid1 < st->num_gates);
  assert(gid2 < st->num_gates || type == NOT);
  assert(gid1 != gid2);
  if (gid1 == NO_GATE || (gid2 == NO_GATE && type != NOT)) {
    return NO_GATE;
  }
  if (st->num_gates > st->max_gates) {
    return NO_GATE;
  }
  if (opt->metric == SAT && st->sat_metric > st->max_sat_metric) {
    return NO_GATE;
  }

  st->sat_metric += get_sat_metric(type);
  if (type == NOT) {
      st->gates[st->num_gates].table = ~st->gates[gid1].table;
  } else {
    st->gates[st->num_gates].table = generate_ttable_2(type, st->gates[gid1].table,
        st->gates[gid2].table);
  }
  st->gates[st->num_gates].type = type;
  st->gates[st->num_gates].in1 = gid1;
  st->gates[st->num_gates].in2 = gid2;
  st->gates[st->num_gates].in3 = NO_GATE;
  st->gates[st->num_gates].function = 0;
  st->num_gates += 1;
  return st->num_gates - 1;
}

gatenum add_lut(state *st, uint8_t func, ttable table, gatenum gid1, gatenum gid2, gatenum gid3) {
  if (gid1 == NO_GATE || gid2 == NO_GATE || gid3 == NO_GATE || st->num_gates > st->max_gates) {
    return NO_GATE;
  }
  assert(gid1 < st->num_gates);
  assert(gid2 < st->num_gates);
  assert(gid3 < st->num_gates);
  assert(gid1 != gid2 && gid2 != gid3 && gid3 != gid1);
  st->gates[st->num_gates].table = table;
  st->gates[st->num_gates].type = LUT;
  st->gates[st->num_gates].in1 = gid1;
  st->gates[st->num_gates].in2 = gid2;
  st->gates[st->num_gates].in3 = gid3;
  st->gates[st->num_gates].function = func;
  st->num_gates += 1;
  return st->num_gates - 1;
}

/* The functions below are all calls to add_gate above added to improve code readability. */

static gatenum add_not_gate(state *st, gatenum gid, const options *opt) {
  if (gid == NO_GATE) {
    return NO_GATE;
  }
  return add_gate(st, NOT, gid, NO_GATE, opt);
}

static gatenum add_and_gate(state *st, gatenum gid1, gatenum gid2, const options *opt) {
  if (gid1 == NO_GATE || gid2 == NO_GATE) {
    return NO_GATE;
  }
  if (gid1 == gid2) {
    return gid1;
  }
  return add_gate(st, AND, gid1, gid2, opt);
}

static gatenum add_or_gate(state *st, gatenum gid1, gatenum gid2, const options *opt) {
  if (gid1 == NO_GATE || gid2 == NO_GATE) {
    return NO_GATE;
  }
  if (gid1 == gid2) {
    return gid1;
  }
  return add_gate(st, OR, gid1, gid2, opt);
}

static gatenum add_xor_gate(state *st, gatenum gid1, gatenum gid2, const options *opt) {
  if (gid1 == NO_GATE || gid2 == NO_GATE) {
    return NO_GATE;
  }
  return add_gate(st, XOR, gid1, gid2, opt);
}

static gatenum add_boolfunc_2(state * restrict st, const boolfunc * restrict fun, gatenum gid1,
    gatenum gid2, const options * restrict opt) {
  assert(fun->num_inputs == 2);
  if (gid1 == NO_GATE || gid2 == NO_GATE || st->num_gates > st->max_gates) {
    return NO_GATE;
  }
  if (opt->metric == SAT && st->sat_metric > st->max_sat_metric) {
    return NO_GATE;
  }
  if (fun->not_a) {
    gid1 = add_not_gate(st, gid1, opt);
  }
  if (fun->not_b) {
    gid2 = add_not_gate(st, gid2, opt);
  }
  gatenum gid = add_gate(st, fun->fun1, gid1, gid2, opt);
  if (fun->not_out) {
    gid = add_not_gate(st, gid, opt);
  }
  return gid;
}

static gatenum add_boolfunc_3(state * restrict st, const boolfunc * restrict fun, gatenum gid1,
    gatenum gid2, gatenum gid3, const options * restrict opt) {
  if (gid1 == NO_GATE || gid2 == NO_GATE || (gid3 == NO_GATE && fun->num_inputs == 3)
      || st->num_gates > st->max_gates) {
    return NO_GATE;
  }
  if (opt->metric == SAT && st->sat_metric > st->max_sat_metric) {
    return NO_GATE;
  }
  if (fun->not_a) {
    gid1 = add_not_gate(st, gid1, opt);
  }
  if (fun->not_b) {
    gid2 = add_not_gate(st, gid2, opt);
  }
  if (fun->not_c) {
    gid3 = add_not_gate(st, gid3, opt);
  }
  gatenum out1 = add_gate(st, fun->fun1, gid1, gid2, opt);
  if (fun->not_out) {
    return add_not_gate(st, add_gate(st, fun->fun2, out1, gid3, opt), opt);
  }
  return add_gate(st, fun->fun2, out1, gid3, opt);
}

/* Returns the number of outputs in the current target S-box. */
static int get_num_outputs() {
  static int outputs = -1;
  if (outputs != -1) {
    return outputs;
  }
  for (int i = 7; i >= 0; i--) {
    if (!ttable_zero(g_target[i])) {
      outputs = i + 1;
      return outputs;
    }
  }
  assert(0);
}

uint64_t xorshift1024() {
  static bool init = false;
  static uint64_t rand[16];
  static int p = 0;
  if (!init) {
    FILE *rand_fp = fopen("/dev/urandom", "r");
    if (rand_fp == NULL) {
      fprintf(stderr, "Error opening /dev/urandom. (sboxgates.c:%d)\n", __LINE__);
    } else if (fread(rand, 16 * sizeof(uint64_t), 1, rand_fp) != 1) {
      fprintf(stderr, "Error reading from /dev/urandom. (sboxgates.c:%d)\n", __LINE__);
      fclose(rand_fp);
    } else {
      init = true;
      fclose(rand_fp);
    }
  }
  uint64_t r0 = rand[p];
  p = (p + 1) & 15;
  uint64_t r1 = rand[p];
  r1 ^= r1 << 31;
  rand[p] = r1 ^ r0 ^ (r1 >> 11) ^ (r0 >> 30);
  return rand[p] * 1181783497276652981U;
}

bool check_num_gates_possible(const state *st, int add, int add_sat, const options *opt) {
  if (opt->metric == SAT && st->sat_metric + add_sat > st->max_sat_metric) {
    return false;
  }
  if (st->num_gates + add > st->max_gates) {
    return false;
  }
  return true;
}

/* Recursively builds the gate network. The numbered comments are references to Matthew Kwan's
   paper. */
static gatenum create_circuit(state *st, const ttable target, const ttable mask,
    const int8_t *inbits, const options *opt) {

  gatenum gate_order[MAX_GATES];
  for (int i = 0; i < st->num_gates; i++) {
    gate_order[i] = st->num_gates - 1 - i;
  }

  /* Randomize the gate search order. */
  if (opt->randomize) {
    /* Fisher-Yates shuffle. */
    for (uint32_t i = st->num_gates - 1; i > 0; i--) {
      uint64_t j = xorshift1024() % (i + 1);
      gatenum t = gate_order[i];
      gate_order[i] = gate_order[j];
      gate_order[j] = t;
    }
  }

  /* 1. Look through the existing circuit. If there is a gate that produces the desired map, simply
     return the ID of that gate. */

  for (int i = 0; i < st->num_gates; i++) {
    if (ttable_equals_mask(target, st->gates[gate_order[i]].table, mask)) {
      ASSERT_AND_RETURN(gate_order[i], target, st, mask);
    }
  }

  /* 2. If there are any gates whose inverse produces the desired map, append a NOT gate, and
     return the ID of the NOT gate. */

  if (!check_num_gates_possible(st, 1, get_sat_metric(NOT), opt)) {
    return NO_GATE;
  }

  for (int i = 0; i < st->num_gates; i++) {
    if (ttable_equals_mask(target, ~st->gates[gate_order[i]].table, mask)) {
      ASSERT_AND_RETURN(add_not_gate(st, gate_order[i], opt), target, st, mask);
    }
  }

  /* 3. Look at all pairs of gates in the existing circuit. If they can be combined with a single
     gate to produce the desired map, add that single gate and return its ID. */

  if (!check_num_gates_possible(st, 1, get_sat_metric(AND), opt)) {
    return NO_GATE;
  }

  const ttable mtarget = target & mask;
  for (int i = 0; i < st->num_gates; i++) {
    const gatenum gi = gate_order[i];
    const ttable ti = st->gates[gi].table;
    for (int k = i + 1; k < st->num_gates; k++) {
      const gatenum gk = gate_order[k];
      const ttable tk = st->gates[gk].table;
      for (int m = 0; opt->avail_gates[m].num_inputs != 0; m++) {
        if (ttable_equals(mtarget, generate_ttable_2(opt->avail_gates[m].fun, ti, tk))) {
          ASSERT_AND_RETURN(add_boolfunc_2(st, &opt->avail_gates[m], gi, gk, opt), target, st,
              mask);
        }
        if (!opt->avail_gates[m].ab_commutative) {
          if (ttable_equals(mtarget, generate_ttable_2(opt->avail_gates[m].fun, tk, ti))) {
            ASSERT_AND_RETURN(add_boolfunc_2(st, &opt->avail_gates[m], gk, gi, opt), target, st,
                mask);
          }
        }
      }
    }
  }

  if (opt->lut_graph) {
    gatenum ret = lut_search(st, target, mask, inbits, gate_order, opt);
    if (ret != NO_GATE) {
      ASSERT_AND_RETURN(ret, target, st, mask);
    }
  } else {
    /* 4. Look at all combinations of two or three gates in the circuit. If they can be combined
       with two gates to produce the desired map, add the gates, and return the ID of the one that
       produces the desired map. */

    if (!check_num_gates_possible(st, 2, get_sat_metric(AND) + get_sat_metric(NOT), opt)) {
      return NO_GATE;
    }

    /* All combinations of two gates. */
    for (int i = 0; i < st->num_gates; i++) {
      const gatenum gi = gate_order[i];
      ttable ti = st->gates[gi].table;
      for (int k = i + 1; k < st->num_gates; k++) {
        const gatenum gk = gate_order[k];
        ttable tk = st->gates[gk].table;
        for (int m = 0; opt->avail_not[m].num_inputs != 0; m++) {
          if (ttable_equals(mtarget, generate_ttable_2(opt->avail_not[m].fun, ti, tk))) {
            ASSERT_AND_RETURN(add_boolfunc_2(st, &opt->avail_not[m], gi, gk, opt), target, st,
                mask);
          }
          if (!opt->avail_not[m].ab_commutative) {
            if (ttable_equals(mtarget, generate_ttable_2(opt->avail_not[m].fun, tk, ti))) {
              ASSERT_AND_RETURN(add_boolfunc_2(st, &opt->avail_not[m], gk, gi, opt), target, st,
                  mask);
            }
          }
        }
      }
    }

    if (!check_num_gates_possible(st, 3, 2 * get_sat_metric(AND) + get_sat_metric(NOT), opt)) {
      return NO_GATE;
    }

    /* All combinations of three gates. */
    for (int i = 0; i < st->num_gates; i++) {
      const gatenum gi = gate_order[i];
      ttable ti = st->gates[gi].table;
      for (int k = i + 1; k < st->num_gates; k++) {
        const gatenum gk = gate_order[k];
        ttable tk = st->gates[gk].table;
        for (int m = k + 1; m < st->num_gates; m++) {
          const gatenum gm = gate_order[m];
          ttable tm = st->gates[gm].table;
          const ttable tables[] = {ti, tk, tm};
          if (!check_n_lut_possible(3, target, mask, tables)) {
            continue;
          }
          for (int p = 0; opt->avail_3[p].num_inputs != 0; p++) {
            if (ttable_equals_mask(target, generate_ttable_3(opt->avail_3[p], ti, tk, tm), mask)) {
              ASSERT_AND_RETURN(add_boolfunc_3(st, &opt->avail_3[p], gi, gk, gm, opt), target, st,
                  mask);
            }
            if (!opt->avail_3[m].ab_commutative) {
              if (ttable_equals_mask(target, generate_ttable_3(opt->avail_3[p], tk, ti, tm),
                  mask)) {
                ASSERT_AND_RETURN(add_boolfunc_3(st, &opt->avail_3[p], gk, gi, gm, opt), target, st,
                    mask);
              }
            }
            if (!opt->avail_3[m].ac_commutative) {
              if (ttable_equals_mask(target, generate_ttable_3(opt->avail_3[p], tm, tk, ti),
                  mask)) {
                ASSERT_AND_RETURN(add_boolfunc_3(st, &opt->avail_3[p], gm, gk, gi, opt), target, st,
                    mask);
              }
            }
            if (!opt->avail_3[m].bc_commutative) {
              if (ttable_equals_mask(target, generate_ttable_3(opt->avail_3[p], ti, tm, tk),
                  mask)) {
                ASSERT_AND_RETURN(add_boolfunc_3(st, &opt->avail_3[p], gi, gm, gk, opt), target, st,
                    mask);
              }
            }
          }
        }
      }
    }
  } /* End of if (opt->lut_graph)... */

  /* 5. Use the specified input bit to select between two Karnaugh maps. Call this function
     recursively to generate those two maps. */

  /* Copy input bits already used to new array to avoid modifying the old one. */
  int8_t next_inbits[8];
  uint8_t bitp = 0;
  while (bitp < 6 && inbits[bitp] != -1) {
    next_inbits[bitp] = inbits[bitp];
    bitp += 1;
  }
  assert(bitp < 7);
  next_inbits[bitp] = -1;
  next_inbits[bitp + 1] = -1;

  state best;
  gatenum best_out = NO_GATE;
  best.num_gates = 0;
  best.sat_metric = 0;

  /* Try all input bit orders. */
  for (int bit = 0; bit < get_num_inputs(st); bit++) {
    /* Skip input bits that have already been used for multiplexing. */
    bool skip = false;
    for (int i = 0; i < bitp; i++) {
      if (inbits[i] == bit) {
        skip = true;
        break;
      }
    }
    if (skip == true) {
      continue;
    }
    next_inbits[bitp] = bit;

    const ttable fsel = st->gates[bit].table; /* Selection bit. */
    state nst;
    gatenum nst_out;
    if (opt->lut_graph) { /* Use a LUT-based multiplexer. */
      nst = *st;
      nst.max_gates -= 1; /* A multiplexer will have to be added later. */
      gatenum fb = create_circuit(&nst, target, mask & ~fsel, next_inbits, opt);
      if (fb == NO_GATE) {
        continue;
      }
      assert(ttable_equals_mask(target, nst.gates[fb].table, mask & ~fsel));
      gatenum fc = create_circuit(&nst, target, mask & fsel, next_inbits, opt);
      if (fc == NO_GATE) {
        continue;
      }
      assert(ttable_equals_mask(target, nst.gates[fc].table, mask & fsel));
      nst.max_gates += 1;

      if (fb == fc) {
        nst_out = fb;
        assert(ttable_equals_mask(target, nst.gates[nst_out].table, mask));
      } else if (fb == bit) {
        nst_out = add_and_gate(&nst, fb, fc, opt);
        if (nst_out == NO_GATE) {
          continue;
        }
        assert(ttable_equals_mask(target, nst.gates[nst_out].table, mask));
      } else if (fc == bit) {
        nst_out = add_or_gate(&nst, fb, fc, opt);
        if (nst_out == NO_GATE) {
          continue;
        }
        assert(ttable_equals_mask(target, nst.gates[nst_out].table, mask));
      } else {
        ttable mux_table = generate_lut_ttable(0xac, nst.gates[bit].table, nst.gates[fb].table,
            nst.gates[fc].table);
        nst_out = add_lut(&nst, 0xac, mux_table, bit, fb, fc);
        if (nst_out == NO_GATE) {
          continue;
        }
        assert(ttable_equals_mask(target, nst.gates[nst_out].table, mask));
      }
      assert(ttable_equals_mask(target, nst.gates[nst_out].table, mask));
    } else { /* Not a LUT graph. Test both AND- and OR-based multiplexers. */
      state nst_and = *st; /* New state using AND multiplexer. */

      /* A multiplexer will have to be added later. */
      nst_and.max_gates -= 2;
      nst_and.max_sat_metric -= get_sat_metric(AND) + get_sat_metric(XOR);

      gatenum fb = create_circuit(&nst_and, target & ~fsel, mask & ~fsel, next_inbits, opt);
      assert(fb == NO_GATE || ttable_equals_mask(target, nst_and.gates[fb].table, mask & ~fsel));
      gatenum mux_out_and = NO_GATE;
      if (fb != NO_GATE) {
        gatenum fc = create_circuit(&nst_and, nst_and.gates[fb].table ^ target, mask & fsel,
            next_inbits, opt);
        assert(fc == NO_GATE || ttable_equals_mask(nst_and.gates[fb].table ^ target,
            nst_and.gates[fc].table, mask & fsel));
        /* Add back subtracted max from above. */
        nst_and.max_gates += 2;
        nst_and.max_sat_metric += get_sat_metric(AND) + get_sat_metric(XOR);
        gatenum andg = add_and_gate(&nst_and, fc, bit, opt);
        mux_out_and = add_xor_gate(&nst_and, fb, andg, opt);
        assert(mux_out_and == NO_GATE ||
            ttable_equals_mask(target, nst_and.gates[mux_out_and].table, mask));
      }

      state nst_or = *st; /* New state using OR multiplexer. */
      if (mux_out_and != NO_GATE) {
        nst_or.max_gates = nst_and.num_gates;
        nst_or.max_sat_metric = nst_and.sat_metric;
      }

      /* A multiplexer will have to be added later. */
      nst_or.max_gates -= 2;
      nst_or.max_sat_metric -= get_sat_metric(OR) + get_sat_metric(XOR);

      gatenum fd = create_circuit(&nst_or, ~target & fsel, mask & fsel, next_inbits, opt);
      assert(fd == NO_GATE || ttable_equals_mask(~target & fsel, nst_or.gates[fd].table,
          mask & fsel));
      gatenum mux_out_or = NO_GATE;
      if (fd != NO_GATE) {
        gatenum fe = create_circuit(&nst_or, nst_or.gates[fd].table ^ target, mask & ~fsel,
            next_inbits, opt);
        assert(fe == NO_GATE || ttable_equals_mask(nst_or.gates[fd].table ^ target,
            nst_or.gates[fe].table, mask & ~fsel));
        /* Add back subtracted max from above. */
        nst_or.max_gates += 2;
        nst_or.max_sat_metric += get_sat_metric(AND) + get_sat_metric(XOR);
        gatenum org = add_or_gate(&nst_or, fe, bit, opt);
        mux_out_or = add_xor_gate(&nst_or, fd, org, opt);
        assert(mux_out_or == NO_GATE ||
            ttable_equals_mask(target, nst_or.gates[mux_out_or].table, mask));
        nst_or.max_gates = st->max_gates;
        nst_or.max_sat_metric = st->max_sat_metric;
      }
      if (mux_out_and == NO_GATE && mux_out_or == NO_GATE) {
        continue;
      }

      if (opt->metric == GATES) {
        if (mux_out_or == NO_GATE
            || (mux_out_and != NO_GATE && nst_and.num_gates < nst_or.num_gates)) {
          nst = nst_and;
          nst_out = mux_out_and;
        } else {
          nst = nst_or;
          nst_out = mux_out_or;
        }
      } else {
        if (mux_out_or == NO_GATE
            || (mux_out_and != NO_GATE && nst_and.sat_metric < nst_or.sat_metric)) {
          nst = nst_and;
          nst_out = mux_out_and;
        } else {
          nst = nst_or;
          nst_out = mux_out_or;
        }
      }
    } /* End of if (opt->lut_graph)... New state in nst. */

    /* Compare nst to best. */
    assert(best.num_gates == 0 || ttable_equals_mask(target, best.gates[best_out].table, mask));
    if (opt->metric == GATES) {
      if (best.num_gates == 0 || nst.num_gates < best.num_gates) {
        best = nst;
        best_out = nst_out;
      }
    } else {
      if (best.sat_metric == 0 || nst.sat_metric < best.sat_metric) {
        best = nst;
        best_out = nst_out;
      }
    }
    assert(best.num_gates == 0 || ttable_equals_mask(target, best.gates[best_out].table, mask));
  } /* End of for loop over all input bits. */

  if (best.num_gates == 0) {
    return NO_GATE;
  }

  assert(ttable_equals_mask(target, best.gates[best_out].table, mask));
  *st = best;
  return best_out;
}

/* All MPI ranks except rank 0 will call this function and wait for work units. */
static void mpi_worker() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  uint16_t res[10];
  while (1) {
    mpi_work work;
    MPI_Bcast(&work, 1, g_mpi_work_type, 0, MPI_COMM_WORLD);
    if (work.quit) {
      return;
    }

    if (work.st.num_gates >= 5
        && search_5lut(work.st, work.target, work.mask, work.inbits, res, work.verbosity)) {
      continue;
    }
    bool search7;
    MPI_Bcast(&search7, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (search7 && work.st.num_gates >= 7) {
      search_7lut(work.st, work.target, work.mask, work.inbits, res, work.verbosity);
    }
  }
}

static ttable generate_mask(int num_inputs) {
  uint64_t mask_vec[] = {0xFFFFFFFFFFFFFFFFUL, 0xFFFFFFFFFFFFFFFFUL,
                         0xFFFFFFFFFFFFFFFFUL, 0xFFFFFFFFFFFFFFFFUL};
  if (num_inputs < 8) {
    mask_vec[2] = mask_vec[3] = 0;
  }
  if (num_inputs < 7) {
    mask_vec[1] = 0;
  }
  if (num_inputs < 6) {
    mask_vec[0] = (1L << (1 << num_inputs)) - 1;
  }
  ttable t;
  memcpy(&t, &mask_vec, sizeof(ttable));
  return t;
}

void generate_graph_one_output(state st, const options *opt) {
  assert(opt->iterations > 0);
  assert(opt->oneoutput >= 0 && opt->oneoutput <= get_num_outputs() - 1);
  printf("Generating graphs for output %d...\n", opt->oneoutput);
  for (int iter = 0; iter < opt->iterations; iter++) {
    state nst = st;

    int8_t bits[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    const ttable mask = generate_mask(get_num_inputs(&st));
    nst.outputs[opt->oneoutput] = create_circuit(&nst, g_target[opt->oneoutput], mask, bits, opt);
    if (nst.outputs[opt->oneoutput] == NO_GATE) {
      printf("(%d/%d): Not found.\n", iter + 1, opt->iterations);
      continue;
    }
    printf("(%d/%d): %d gates. SAT metric: %d\n", iter + 1, opt->iterations,
        nst.num_gates - get_num_inputs(&nst), nst.sat_metric);
    save_state(nst);
    if (opt->metric == GATES) {
      if (nst.num_gates < st.max_gates) {
        st.max_gates = nst.num_gates;
      }
    } else {
      if (nst.sat_metric < st.max_sat_metric) {
        st.max_sat_metric = nst.sat_metric;
      }
    }
  }
}

static inline int count_state_outputs(state st) {
  int num_outputs = 0;
  for (int i = 0; i < 8; i++) {
    if (st.outputs[i] != NO_GATE) {
      num_outputs += 1;
    }
  }
  return num_outputs;
}

/* Called by main to generate a graph. */
void generate_graph(const state st, const options *opt) {
  assert(opt != NULL);
  int num_start_states = 1;
  state start_states[20];
  start_states[0] = st;

  /* Build the gate network one output at a time. After every added output, select the gate network
     or network with the least amount of gates and add another. */
  int num_outputs;
  while ((num_outputs = count_state_outputs(start_states[0])) < get_num_outputs()) {
    gatenum max_gates = MAX_GATES;
    int max_sat_metric = INT_MAX;
    state out_states[20];
    memset(out_states, 0, sizeof(state) * 20);
    int num_out_states = 0;

    for (int iter = 0; iter < opt->iterations; iter++) {
      printf("Generating circuits with %d output%s. (%d/%d)\n", num_outputs + 1,
          num_outputs == 0 ? "" : "s", iter + 1, opt->iterations);
      for (uint8_t current_state = 0; current_state < num_start_states; current_state++) {
        start_states[current_state].max_gates = max_gates;
        start_states[current_state].max_sat_metric = max_sat_metric;

        /* Add all outputs not already present to see which resulting network is the smallest. */
        for (uint8_t output = 0; output < get_num_outputs(); output++) {
          if (start_states[current_state].outputs[output] != NO_GATE) {
            printf("Skipping output %d.\n", output);
            continue;
          }
          printf("Generating circuit for output %d...\n", output);
          int8_t bits[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
          state st = start_states[current_state];
          if (opt->metric == GATES) {
            st.max_gates = max_gates;
          } else {
            st.max_sat_metric = max_sat_metric;
          }

          const ttable mask = generate_mask(get_num_inputs(&st));
          st.outputs[output] = create_circuit(&st, g_target[output], mask, bits, opt);
          if (st.outputs[output] == NO_GATE) {
            printf("No solution for output %d.\n", output);
            continue;
          }
          assert(ttable_equals_mask(g_target[output], st.gates[st.outputs[output]].table, mask));
          save_state(st);

          if (opt->metric == GATES) {
            if (max_gates > st.num_gates) {
              max_gates = st.num_gates;
              num_out_states = 0;
            }
            if (st.num_gates <= max_gates) {
              if (num_out_states < 20) {
                out_states[num_out_states++] = st;
              } else {
                printf("Output state buffer full! Throwing away valid state.\n");
              }
            }
          } else {
            if (max_sat_metric > st.sat_metric) {
              max_sat_metric = st.sat_metric;
              num_out_states = 0;
            }
            if (st.sat_metric <= max_sat_metric) {
              if (num_out_states < 20) {
                out_states[num_out_states++] = st;
              } else {
                printf("Output state buffer full! Throwing away valid state.\n");
              }
            }
          }
        }
      }
    }
    if (opt->metric == GATES) {
      printf("Found %d state%s with %d gates.\n", num_out_states,
          num_out_states == 1 ? "" : "s", max_gates - get_num_inputs(&out_states[0]));
    } else {
      printf("Found %d state%s with SAT metric %d.\n", num_out_states,
          num_out_states == 1 ? "" : "s", max_sat_metric);
    }
    for (int i  = 0; i < num_out_states; i++) {
      start_states[i] = out_states[i];
    }
    num_start_states = num_out_states;
  }
}

/* Causes the MPI workers to quit. */
static void stop_workers() {
  mpi_work work;
  work.quit = true;
  MPI_Bcast(&work, 1, g_mpi_work_type, 0, MPI_COMM_WORLD);
}

/* Called by main to create data types for structures passed between MPI instances. */
void create_g_mpi_work_type() {
  /* gate struct */
  int gate_block_lengths[] = {4, 1, 1, 1, 1, 1};
  MPI_Aint gate_displacements[] = {
      offsetof(gate, table),
      offsetof(gate, type),
      offsetof(gate, in1),
      offsetof(gate, in2),
      offsetof(gate, in3),
      offsetof(gate, function)
    };
  MPI_Datatype gate_datatypes[] = {
      MPI_UINT64_T,
      MPI_INT,
      MPI_UINT16_T,
      MPI_UINT16_T,
      MPI_UINT16_T,
      MPI_UINT8_T
    };
  MPI_Datatype gate_type;
  assert(MPI_Type_create_struct(6, gate_block_lengths, gate_displacements, gate_datatypes,
        &gate_type) == MPI_SUCCESS);
  assert(MPI_Type_create_resized(gate_type, 0, sizeof(gate), &gate_type)
      == MPI_SUCCESS);
  assert(MPI_Type_commit(&gate_type) == MPI_SUCCESS);

  /* state struct */
  int state_block_lengths[] = {1, 1, 1, 1, 8, MAX_GATES};
  MPI_Aint state_displacements[] = {
      offsetof(state, max_sat_metric),
      offsetof(state, sat_metric),
      offsetof(state, max_gates),
      offsetof(state, num_gates),
      offsetof(state, outputs),
      offsetof(state, gates)
    };
  MPI_Datatype state_datatypes[] = {
      MPI_INT,
      MPI_INT,
      MPI_UINT16_T,
      MPI_UINT16_T,
      MPI_UINT16_T,
      gate_type
    };
  MPI_Datatype state_type;
  assert(MPI_Type_create_struct(6, state_block_lengths, state_displacements, state_datatypes,
      &state_type) == MPI_SUCCESS);
  assert(MPI_Type_commit(&state_type) == MPI_SUCCESS);

  /* mpi_work struct*/
  int work_block_lengths[] = {1, 4, 4, 8, 1, 1};
  MPI_Aint work_displacements[] = {
      offsetof(mpi_work, st),
      offsetof(mpi_work, target),
      offsetof(mpi_work, mask),
      offsetof(mpi_work, inbits),
      offsetof(mpi_work, quit),
      offsetof(mpi_work, verbosity)
    };
  MPI_Datatype work_datatypes[] = {
      state_type,
      MPI_UINT64_T,
      MPI_UINT64_T,
      MPI_UINT8_T,
      MPI_C_BOOL,
      MPI_INT
    };
  assert(MPI_Type_create_struct(6, work_block_lengths, work_displacements, work_datatypes,
      &g_mpi_work_type) == MPI_SUCCESS);
  assert(MPI_Type_commit(&g_mpi_work_type) == MPI_SUCCESS);
}

static void create_avail_gates(uint16_t gates, options *opt) {
  assert(opt != NULL);
  opt->avail_gates[0].num_inputs = 0;
  int gatep = 0;
  for (int i = 0; i < 16; i++) {
    if (gates & (1 << i)) {
      opt->avail_gates[gatep++] = create_2_input_fun(i);
      opt->avail_gates[gatep].num_inputs = 0;
    }
  }
}

/* Used in parse_opt to increase readability. */
#define PARSE_OPTIONS_EXIT()\
  stop_workers();\
  MPI_Finalize();\
  exit(1);
#define PARSE_OPTIONS_TEST_NAME_LENGTH(X)\
  if (strlen(X) >= MAX_NAME_LEN) {\
    fprintf(stderr, "Error: File name too long. (sboxgates.c:%d)\n", __LINE__);\
    stop_workers();\
    MPI_Finalize();\
    exit(1);\
  }

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
  options *opt = state->input;
  int avail_gates;
  char *endptr;
  switch (key) {
    case 'a':
      avail_gates = atoi(arg);
      if (avail_gates <= 0 || avail_gates > 65535) {
        fprintf(stderr, "Bad available gates value: %s (sboxgates.c:%d)\n", arg, __LINE__);
        PARSE_OPTIONS_EXIT();
      }
      create_avail_gates(avail_gates, opt);
      return 0;
    case 'c':
      opt->output_c = true;
      return 0;
    case 'd':
      opt->output_dot = true;
      return 0;
    case 'g':
      PARSE_OPTIONS_TEST_NAME_LENGTH(arg);
      strcpy(opt->gfname, arg);
      return 0;
    case 'i':
      opt->iterations = strtoul(arg, &endptr, 10);
      if (*endptr != '\0' || opt->iterations < 1) {
        fprintf(stderr, "Bad iterations value: %s (sboxgates.c:%d)\n", arg, __LINE__);
        PARSE_OPTIONS_EXIT();
      }
      return 0;
    case 'l':
      opt->lut_graph = true;
      return 0;
    case 'n':
      opt->try_nots = true;
      return 0;
    case 'o':
      opt->oneoutput = strtoul(arg, &endptr, 10);
      if (*endptr != '\0' || opt->oneoutput < 0 || opt->oneoutput > 7) {
        fprintf(stderr, "Bad output value: %s (sboxgates.c:%d)\n", arg, __LINE__);
        PARSE_OPTIONS_EXIT();
      }
      return 0;
    case 'p':
      opt->permute = strtoul(arg, &endptr, 10);
      if (*endptr != '\0' || opt->permute < 0 || opt->permute > 255) {
        fprintf(stderr, "Bad permutation value: %s (sboxgates.c:%d)\n", arg, __LINE__);
        PARSE_OPTIONS_EXIT();
      }
      return 0;
    case 's':
      opt->metric = SAT;
      return 0;
    case 'v':
      opt->verbosity += 1;
      return 0;
    case ARGP_KEY_ARG:
      if (strlen(opt->fname) != 0) {
        return 0;
      }
      PARSE_OPTIONS_TEST_NAME_LENGTH(arg);
      strcpy(opt->fname, arg);
      return 0;
    case ARGP_KEY_END:
      if (opt->output_c && opt->output_dot) {
        fprintf(stderr, "Cannot combine c and d options. (sboxgates.c:%d)\n", __LINE__);
        PARSE_OPTIONS_EXIT();
      }

      if (opt->lut_graph && opt->metric == SAT) {
        fprintf(stderr, "SAT metric can not be combined with LUT graph generation. "
            "(sboxgates.c:%d)\n", __LINE__);
        PARSE_OPTIONS_EXIT();
      }

      if (strlen(opt->fname) == 0) {
        fprintf(stderr, "Input file name argument missing. (sboxgates.c:%d)\n", __LINE__);
        PARSE_OPTIONS_EXIT();
      }
      /* Create derived boolean functions. */
      int num = 0;
      if (opt->try_nots) {
        num = get_not_functions(opt->avail_gates, opt->avail_not);
      }
      memset(opt->avail_not + num, 0, sizeof(boolfunc));
      num = get_3_input_function_list(opt->avail_gates, opt->avail_3, opt->try_nots);
      memset(opt->avail_3 + num, 0, sizeof(boolfunc));
      return 0;
    default:
      return ARGP_ERR_UNKNOWN;
  }
}

/* Loads an S-box from a file. The file should contain the S-box table as 2^n (1 <= n <= 8)
   whitespace separated hexadecimal numbers. The S-box is loaded into the 256 item array pointed to
   by sbox and num_input is set to the calculated number of input bits. The input file name is
   taken from the opt structure. */
bool load_sbox(uint8_t *sbox, uint32_t *num_inputs, const options *opt) {
  assert(sbox != NULL);
  assert(num_inputs != NULL);
  assert(opt != NULL);
  assert(opt->fname != NULL);
  int sbox_inp = 0;

  FILE *fp = fopen(opt->fname, "r");
  if (fp == NULL) {
    fprintf(stderr, "Error when opening target S-box file. (sboxgates.c:%d)\n", __LINE__);
    return false;
  }

  int ret;
  uint8_t target_sbox[256];
  memset(target_sbox, 0, sizeof(uint8_t) * 256);
  uint32_t input;
  while ((ret = fscanf(fp, " %x", &input)) > 0 && ret != EOF && sbox_inp < 256 && input < 256) {
    target_sbox[sbox_inp++] = input;
  }
  fclose(fp);

  if (__builtin_popcount(sbox_inp) != 1) {
    fprintf(stderr, "Bad number of items in target S-box. (sboxgates.c:%d)\n", __LINE__);
    return false;
  }

  *num_inputs = 31 - __builtin_clz(sbox_inp);

  if (opt->permute == 0) {
    memcpy(sbox, target_sbox, sizeof(uint8_t) * 256);
  } else {
    if (opt->permute >= (1 << *num_inputs)) {
      fprintf(stderr, "Bad permutation value: %d (sboxgates.c:%d)\n", opt->permute, __LINE__);
      return false;
    }
    for (int i = 0; i < 256; i++) {
      sbox[i] = target_sbox[i ^ (uint8_t)opt->permute];
    }
  }

  if (opt->verbosity >= 2) {
    printf("Loaded %d input S-box:\n", *num_inputs);
    for (int i = 0; i < sbox_inp; i++) {
      printf("%02x%s", sbox[i], (i + 1) % 16 ? " " : "\n");
    }
  }
  return true;
}

static struct argp argp = {argp_options, parse_opt, args_doc, doc, 0, 0, 0};

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  create_g_mpi_work_type();

  /* Let all ranks except for rank 0 go into worker loop. */
  if (rank != 0) {
    mpi_worker();
    MPI_Finalize();
    return 0;
  }

  /* Parse command line options. */
  options opt = {
    .fname = {0},
    .gfname = {0},
    .iterations = 1,
    .oneoutput = -1,
    .permute = 0,
    .metric = GATES,
    .output_c = false,
    .output_dot = false,
    .lut_graph = false,
    .randomize = true,
    .try_nots = false,
    .avail_gates = {{0}},
    .avail_not = {{0}},
    .avail_3 = {{0}},
    .num_avail_3 = 0,
    .verbosity = 0
  };
  create_avail_gates(2 + 64 + 128, &opt); /* AND + OR + XOR */
  argp_parse(&argp, argc, argv, 0, 0, &opt);
  if (opt.verbosity >= 1) {
    printf("Available gates: NOT ");
    for (int i = 0; opt.avail_gates[i].num_inputs != 0; i++) {
      printf("%s ", gate_name[opt.avail_gates[i].fun]);
    }
    printf("\nGenerated gates: ");
    for (int i = 0; opt.avail_not[i].num_inputs != 0; i++) {
      printf("%s ", gate_name[opt.avail_not[i].fun]);
    }
    printf("\nGenerated 3-input gates: ");
    for (int i = 0; opt.avail_3[i].num_inputs != 0; i++) {
      printf("%02x ", opt.avail_3[i].fun);
    }
    printf("\n");
  }

  /* Convert graph to C or DOT output and quit. */
  if (opt.output_c || opt.output_dot) {
    stop_workers();
    state st;
    if (!load_state(opt.fname, &st)) {
      fprintf(stderr, "Error when reading state file. (sboxgates.c:%d)\n", __LINE__);
      MPI_Finalize();
      return 1;
    }
    int retval = 0;
    if (opt.output_c) {
      retval = print_c_function(&st) ? 0 : 1;
    } else {
      print_digraph(&st);
    }
    MPI_Finalize();
    return retval;
  }

  /* Load specified S-box from file. */
  uint32_t num_inputs; /* Used to initialize the input gates below. */
  if (!load_sbox(g_sbox_enc, &num_inputs, &opt)) {
    stop_workers();
    MPI_Finalize();
    return 1;
  }

  /* Generate truth tables for all output bits of the target sbox. */
  for (uint8_t i = 0; i < 8; i++) {
    g_target[i] = generate_target(i, true);
  }

  if (opt.oneoutput >= get_num_outputs()) {
    fprintf(stderr, "Error: Can't generate output bit %d. Target S-box only has %d outputs. "
        "(sboxgates.c:%d)\n", opt.oneoutput, get_num_outputs(), __LINE__);
    stop_workers();
    MPI_Finalize();
    return 1;
  }

  /* Initialize the state structure. */
  state st;
  memset(&st, 0, sizeof(state));
  if (strlen(opt.gfname) == 0) {
    st.max_sat_metric = INT_MAX;
    st.sat_metric = 0;
    st.max_gates = MAX_GATES;
    st.num_gates = num_inputs;
    for (int i = 0; i < num_inputs; i++) {
      st.gates[i].type = IN;
      st.gates[i].table = generate_target(i, false);
      st.gates[i].in1 = NO_GATE;
      st.gates[i].in2 = NO_GATE;
      st.gates[i].in3 = NO_GATE;
      st.gates[i].function = 0;
    }
    for (int i = 0; i < 8; i++) {
      st.outputs[i] = NO_GATE;
    }
  } else if (!load_state(opt.gfname, &st)) {
    stop_workers();
    MPI_Finalize();
    return 1;
  } else {
    printf("Loaded %s.\n", opt.gfname);
  }

  /* Generate the graph. */
  if (opt.oneoutput != -1) {
    generate_graph_one_output(st, &opt);
  } else {
    generate_graph(st, &opt);
  }

  stop_workers();
  MPI_Finalize();

  return 0;
}
