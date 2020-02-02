/* sboxgates.c

   Program for finding low gate count implementations of S-boxes.
   The algorithm used is described in Kwan, Matthew: "Reducing the Gate Count of Bitslice DES."
   IACR Cryptology ePrint Archive 2000 (2000): 51. Improvements from
   SBOXDiscovery (https://github.com/DeepLearningJohnDoe/SBOXDiscovery) have been added.

   Copyright (c) 2016-2017, 2019-2020 Marcus Dansarie

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

uint8_t g_sbox_enc[256];      /* Target S-box. */
ttable g_target[8];           /* Truth tables for the output bits of the sbox. */

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
static inline gatenum add_gate(state *st, gate_type type, ttable table, gatenum gid1,
    gatenum gid2, const options *opt) {
  if (gid1 == NO_GATE || (gid2 == NO_GATE && type != NOT) || st->num_gates > st->max_gates) {
    return NO_GATE;
  }
  if (opt->metric == SAT && st->sat_metric > st->max_sat_metric) {
    return NO_GATE;
  }
  assert(type != IN && type != LUT);
  assert(gid1 < st->num_gates);
  assert(gid2 < st->num_gates || type == NOT);
  assert(gid1 != gid2);
  st->sat_metric += get_sat_metric(type);
  st->gates[st->num_gates].table = table;
  st->gates[st->num_gates].type = type;
  st->gates[st->num_gates].in1 = gid1;
  st->gates[st->num_gates].in2 = gid2;
  st->gates[st->num_gates].in3 = NO_GATE;
  st->gates[st->num_gates].function = 0;
  st->num_gates += 1;
  return st->num_gates - 1;
}

/* Adds a three input LUT with function func to the state st. Returns the gate number of the
   added LUT. */
gatenum add_lut(state *st, uint8_t func, ttable table, gatenum gid1, gatenum gid2, gatenum gid3) {
  if (gid1 == NO_GATE || gid2 == NO_GATE || gid3 == NO_GATE || st->num_gates > st->max_gates) {
    return NO_GATE;
  }
  assert(gid1 < st->num_gates);
  assert(gid2 < st->num_gates);
  assert(gid3 < st->num_gates);
  if (gid1 == gid2 || gid2 == gid3 || gid3 == gid1) {
    printf("%d %d %d\n", gid1, gid2, gid3);
  }
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

static inline gatenum add_not_gate(state *st, gatenum gid, const options *opt) {
  if (gid == NO_GATE) {
    return NO_GATE;
  }
  return add_gate(st, NOT, ~st->gates[gid].table, gid, NO_GATE, opt);
}

static inline gatenum add_and_gate(state *st, gatenum gid1, gatenum gid2, const options *opt) {
  if (gid1 == NO_GATE || gid2 == NO_GATE) {
    return NO_GATE;
  }
  if (gid1 == gid2) {
    return gid1;
  }
  return add_gate(st, AND, st->gates[gid1].table & st->gates[gid2].table, gid1, gid2, opt);
}

static inline gatenum add_or_gate(state *st, gatenum gid1, gatenum gid2, const options *opt) {
  if (gid1 == NO_GATE || gid2 == NO_GATE) {
    return NO_GATE;
  }
  if (gid1 == gid2) {
    return gid1;
  }
  return add_gate(st, OR, st->gates[gid1].table | st->gates[gid2].table, gid1, gid2, opt);
}

static inline gatenum add_xor_gate(state *st, gatenum gid1, gatenum gid2, const options *opt) {
  if (gid1 == NO_GATE || gid2 == NO_GATE) {
    return NO_GATE;
  }
  return add_gate(st, XOR, st->gates[gid1].table ^ st->gates[gid2].table, gid1, gid2, opt);
}

static inline gatenum add_andnot_gate(state *st, gatenum gid1, gatenum gid2, const options *opt) {
  if (gid1 == NO_GATE || gid2 == NO_GATE) {
    return NO_GATE;
  }
  return add_gate(st, ANDNOT, ~st->gates[gid1].table & st->gates[gid2].table, gid1, gid2, opt);
}

static inline gatenum add_nand_gate(state *st, gatenum gid1, gatenum gid2, const options *opt) {
  return add_not_gate(st, add_and_gate(st, gid1, gid2, opt), opt);
}

static inline gatenum add_nor_gate(state *st, gatenum gid1, gatenum gid2, const options *opt) {
  return add_not_gate(st, add_or_gate(st, gid1, gid2, opt), opt);
}

static inline gatenum add_xnor_gate(state *st, gatenum gid1, gatenum gid2, const options *opt) {
  return add_not_gate(st, add_xor_gate(st, gid1, gid2, opt), opt);
}

static inline gatenum add_or_not_gate(state *st, gatenum gid1, gatenum gid2, const options *opt) {
  return add_or_gate(st, add_not_gate(st, gid1, opt), gid2, opt);
}

static inline gatenum add_and_not_gate(state *st, gatenum gid1, gatenum gid2, const options *opt) {
  return add_and_gate(st, add_not_gate(st, gid1, opt), gid2, opt);
}

static inline gatenum add_or_3_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3,
    const options *opt) {
  return add_or_gate(st, add_or_gate(st, gid1, gid2, opt), gid3, opt);
}

static inline gatenum add_and_3_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3,
    const options *opt) {
  return add_and_gate(st, add_and_gate(st, gid1, gid2, opt), gid3, opt);
}

static inline gatenum add_xor_3_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3,
    const options *opt) {
  return add_xor_gate(st, add_xor_gate(st, gid1, gid2, opt), gid3, opt);
}

static inline gatenum add_and_or_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3,
    const options *opt) {
  return add_or_gate(st, add_and_gate(st, gid1, gid2, opt), gid3, opt);
}

static inline gatenum add_and_xor_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3,
    const options *opt) {
  return add_xor_gate(st, add_and_gate(st, gid1, gid2, opt), gid3, opt);
}

static inline gatenum add_xor_or_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3,
    const options *opt) {
  return add_or_gate(st, add_xor_gate(st, gid1, gid2, opt), gid3, opt);
}

static inline gatenum add_xor_and_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3,
    const options *opt) {
  return add_and_gate(st, add_xor_gate(st, gid1, gid2, opt), gid3, opt);
}

static inline gatenum add_or_and_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3,
    const options *opt) {
  return add_and_gate(st, add_or_gate(st, gid1, gid2, opt), gid3, opt);
}

static inline gatenum add_or_xor_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3,
    const options *opt) {
  return add_xor_gate(st, add_or_gate(st, gid1, gid2, opt), gid3, opt);
}

static inline gatenum add_andnot_or_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3,
    const options *opt) {
  return add_or_gate(st, add_andnot_gate(st, gid1, gid2, opt), gid3, opt);
}

static inline gatenum add_xor_andnot_a_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3,
    const options *opt) {
  return add_andnot_gate(st, gid3, add_xor_gate(st, gid1, gid2, opt), opt);
}

static inline gatenum add_xor_andnot_b_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3,
    const options *opt) {
  return add_andnot_gate(st, add_xor_gate(st, gid1, gid2, opt), gid3, opt);
}

static inline gatenum add_and_andnot_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3,
    const options *opt) {
  return add_and_gate(st, add_andnot_gate(st, gid1, gid2, opt), gid3, opt);
}

static inline gatenum add_andnot_3_a_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3,
    const options *opt) {
  return add_andnot_gate(st, gid1, add_andnot_gate(st, gid2, gid3, opt), opt);
}

static inline gatenum add_andnot_3_b_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3,
    const options *opt) {
  return add_andnot_gate(st, add_andnot_gate(st, gid1, gid2, opt), gid3, opt);
}

static inline gatenum add_andnot_xor_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3,
    const options *opt) {
  return add_xor_gate(st, add_andnot_gate(st, gid1, gid2, opt), gid3, opt);
}

/* Returns the number of input gates in the state. */
int get_num_inputs(const state *st) {
  int inputs = 0;
  for (int i = 0; st->gates[i].type == IN && i < st->num_gates; i++) {
    inputs += 1;
  }
  return inputs;
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

/* Generates pseudorandom 64 bit strings. Used for randomizing the search process. */
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

/* Used in create_circuit to check if any solutions with smaller metric are possible. Uses either
   the add or the add_sat parameter depending on the current metric in use. Returns true if a
   solution with the provided metric is possible with respect to the value of st->max_gates or
   st->max_sat_metric. */
bool check_num_gates_possible(state *st, int add, int add_sat, const options *opt) {
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
      return gate_order[i];
    }
  }

  /* 2. If there are any gates whose inverse produces the desired map, append a NOT gate, and
     return the ID of the NOT gate. */

  if (!check_num_gates_possible(st, 1, get_sat_metric(NOT), opt)) {
    return NO_GATE;
  }

  for (int i = 0; i < st->num_gates; i++) {
    if (ttable_equals_mask(target, ~st->gates[gate_order[i]].table, mask)) {
      return add_not_gate(st, gate_order[i], opt);
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
    const ttable ti = st->gates[gi].table & mask;
    for (int k = i + 1; k < st->num_gates; k++) {
      const gatenum gk = gate_order[k];
      const ttable tk = st->gates[gk].table & mask;
      if (ttable_equals(mtarget, ti | tk)) {
        return add_or_gate(st, gi, gk, opt);
      }
      if (ttable_equals(mtarget, ti & tk)) {
        return add_and_gate(st, gi, gk, opt);
      }
      if (opt->andnot) {
        if (ttable_equals_mask(target, ~ti & tk, mask)) {
          return add_andnot_gate(st, gi, gk, opt);
        }
        if (ttable_equals_mask(target, ~tk & ti, mask)) {
          return add_andnot_gate(st, gk, gi, opt);
        }
      }
      if (ttable_equals(mtarget, ti ^ tk)) {
        return add_xor_gate(st, gi, gk, opt);
      }
    }
  }

  if (opt->lut_graph) {
    gatenum ret = lut_search(st, target, mask, inbits, gate_order, opt);
    if (ret != NO_GATE) {
      return ret;
    }
  } else {
    /* 4. Look at all combinations of two or three gates in the circuit. If they can be combined
       with two gates to produce the desired map, add the gates, and return the ID of the one that
       produces the desired map. */

    if (!check_num_gates_possible(st, 2, get_sat_metric(AND) + get_sat_metric(NOT), opt)) {
      return NO_GATE;
    }

    for (int i = 0; i < st->num_gates; i++) {
      const gatenum gi = gate_order[i];
      ttable ti = st->gates[gi].table;
      for (int k = i + 1; k < st->num_gates; k++) {
        const gatenum gk = gate_order[k];
        ttable tk = st->gates[gk].table;
        if (ttable_equals_mask(target, ~(ti | tk), mask)) {
          return add_nor_gate(st, gi, gk, opt);
        }
        if (ttable_equals_mask(target, ~(ti & tk), mask)) {
          return add_nand_gate(st, gi, gk, opt);
        }
        if (ttable_equals_mask(target, ~ti | tk, mask)) {
          return add_or_not_gate(st, gi, gk, opt);
        }
        if (ttable_equals_mask(target, ~tk | ti, mask)) {
          return add_or_not_gate(st, gk, gi, opt);
        }
        if (!opt->andnot) {
          if (ttable_equals_mask(target, ~ti & tk, mask)) {
            return add_and_not_gate(st, gi, gk, opt);
          }
          if (ttable_equals_mask(target, ~tk & ti, mask)) {
            return add_and_not_gate(st, gk, gi, opt);
          }
        } else if (ttable_equals_mask(target, ~ti & ~tk, mask)) {
          return add_andnot_gate(st, gi, add_not_gate(st, gk, opt), opt);
        }
        if (ttable_equals_mask(target, ~(ti ^ tk), mask)) {
          return add_xnor_gate(st, gi, gk, opt);
        }
      }
    }

    if (!check_num_gates_possible(st, 3, 2 * get_sat_metric(AND) + get_sat_metric(NOT), opt)) {
      return NO_GATE;
    }

    for (int i = 0; i < st->num_gates; i++) {
      const gatenum gi = gate_order[i];
      ttable ti = st->gates[gi].table & mask;
      for (int k = i + 1; k < st->num_gates; k++) {
        const gatenum gk = gate_order[k];
        ttable tk = st->gates[gk].table & mask;
        ttable iandk = ti & tk;
        ttable iork = ti | tk;
        ttable ixork = ti ^ tk;
        for (int m = k + 1; m < st->num_gates; m++) {
          const gatenum gm = gate_order[m];
          ttable tm = st->gates[gm].table & mask;
          const ttable tables[] = {ti, tk, tm};
          if (!check_n_lut_possible(3, target, mask, tables)) {
            continue;
          }
          if (ttable_equals(mtarget, iandk & tm)) {
            return add_and_3_gate(st, gi, gk, gm, opt);
          }
          if (ttable_equals(mtarget, iandk | tm)) {
            return add_and_or_gate(st, gi, gk, gm, opt);
          }
          if (ttable_equals(mtarget, iork | tm)) {
            return add_or_3_gate(st, gi, gk, gm, opt);
          }
          if (ttable_equals(mtarget, iork & tm)) {
            return add_or_and_gate(st, gi, gk, gm, opt);
          }
          ttable iandm = ti & tm;
          if (ttable_equals(mtarget, iandm | tk)) {
            return add_and_or_gate(st, gi, gm, gk, opt);
          }
          ttable kandm = tk & tm;
          if (ttable_equals(mtarget, kandm | ti)) {
            return add_and_or_gate(st, gk, gm, gi, opt);
          }
          ttable iorm = ti | tm;
          if (ttable_equals(mtarget, iorm & tk)) {
            return add_or_and_gate(st, gi, gm, gk, opt);
          }
          ttable korm = tk | tm;
          if (ttable_equals(mtarget, korm & ti)) {
            return add_or_and_gate(st, gk, gm, gi, opt);
          }
          if (opt->andnot) {
            if (ttable_equals(mtarget, ti | (~tk & tm))) {
              return add_andnot_or_gate(st, gk, gm, gi, opt);
            }
            if (ttable_equals(mtarget, ti | (tk & ~tm))) {
              return add_andnot_or_gate(st, gm, gk, gi, opt);
            }
            if (ttable_equals(mtarget, tm | (~ti & tk))) {
              return add_andnot_or_gate(st, gi, gk, gm, opt);
            }
            if (ttable_equals(mtarget, tm | (ti & ~tk))) {
              return add_andnot_or_gate(st, gk, gi, gm, opt);
            }
            if (ttable_equals(mtarget, tk | (~ti & tm))) {
              return add_andnot_or_gate(st, gi, gm, gk, opt);
            }
            if (ttable_equals(mtarget, tk | (ti & ~tm))) {
              return add_andnot_or_gate(st, gm, gi, gk, opt);
            }
            if (ttable_equals(mtarget, ~ti & tk & tm)) {
              return add_and_andnot_gate(st, gi, gk, gm, opt);
            }
            if (ttable_equals(mtarget, ti & ~tk & tm)) {
              return add_and_andnot_gate(st, gk, gi, gm, opt);
            }
            if (ttable_equals(mtarget, ti & tk & ~tm)) {
              return add_and_andnot_gate(st, gm, gk, gi, opt);
            }
            if (ttable_equals(mtarget, ~ti & ~tk & tm)) {
              return add_andnot_3_a_gate(st, gi, gk, gm, opt);
            }
            if (ttable_equals(mtarget, ~ti & tk & ~tm)) {
              return add_andnot_3_a_gate(st, gi, gm, gk, opt);
            }
            if (ttable_equals(mtarget, ti & ~tk & ~tm)) {
              return add_andnot_3_a_gate(st, gk, gm, gi, opt);
            }
            if (ttable_equals(mtarget, ti & ~(~tk & tm))) {
              return add_andnot_3_b_gate(st, gk, gm, gi, opt);
            }
            if (ttable_equals(mtarget, ti & ~(tk & ~tm))) {
              return add_andnot_3_b_gate(st, gm, gk, gi, opt);
            }
            if (ttable_equals(mtarget, tk & ~(~ti & tm))) {
              return add_andnot_3_b_gate(st, gi, gm, gk, opt);
            }
            if (ttable_equals(mtarget, tk & ~(ti & ~tm))) {
              return add_andnot_3_b_gate(st, gm, gi, gk, opt);
            }
            if (ttable_equals(mtarget, tm & ~(~tk & ti))) {
              return add_andnot_3_b_gate(st, gk, gi, gm, opt);
            }
            if (ttable_equals(mtarget, tm & ~(tk & ~ti))) {
              return add_andnot_3_b_gate(st, gi, gk, gm, opt);
            }
            if (ttable_equals(mtarget, ~ti & (tk ^ tm))) {
              return add_xor_andnot_a_gate(st, gk, gm, gi, opt);
            }
            if (ttable_equals(mtarget, ~tk & (ti ^ tm))) {
              return add_xor_andnot_a_gate(st, gi, gm, gk, opt);
            }
            if (ttable_equals(mtarget, ~tm & (tk ^ ti))) {
              return add_xor_andnot_a_gate(st, gk, gi, gm, opt);
            }
            if (ttable_equals(mtarget, ti & ~(tk ^ tm))) {
              return add_xor_andnot_b_gate(st, gk, gm, gi, opt);
            }
            if (ttable_equals(mtarget, tk & ~(ti ^ tm))) {
              return add_xor_andnot_b_gate(st, gi, gm, gk, opt);
            }
            if (ttable_equals(mtarget, tm & ~(tk ^ ti))) {
              return add_xor_andnot_b_gate(st, gk, gi, gm, opt);
            }
            if (ttable_equals(mtarget, ti ^ (~tk & tm))) {
              return add_andnot_xor_gate(st, gk, gm, gi, opt);
            }
            if (ttable_equals(mtarget, ti ^ (tk & ~tm))) {
              return add_andnot_xor_gate(st, gm, gk, gi, opt);
            }
            if (ttable_equals(mtarget, tk ^ (~ti & tm))) {
              return add_andnot_xor_gate(st, gi, gm, gk, opt);
            }
            if (ttable_equals(mtarget, tk ^ (ti & ~tm))) {
              return add_andnot_xor_gate(st, gm, gi, gk, opt);
            }
            if (ttable_equals(mtarget, tm ^ (~tk & ti))) {
              return add_andnot_xor_gate(st, gk, gi, gm, opt);
            }
            if (ttable_equals(mtarget, tm ^ (tk & ~ti))) {
              return add_andnot_xor_gate(st, gi, gk, gm, opt);
            }
          }
          if (ttable_equals(mtarget, ixork | tm)) {
            return add_xor_or_gate(st, gi, gk, gm, opt);
          }
          if (ttable_equals(mtarget, ixork & tm)) {
            return add_xor_and_gate(st, gi, gk, gm, opt);
          }
          if (ttable_equals(mtarget, iandk ^ tm)) {
            return add_and_xor_gate(st, gi, gk, gm, opt);
          }
          if (ttable_equals(mtarget, iork ^ tm)) {
            return add_or_xor_gate(st, gi, gk, gm, opt);
          }
          if (ttable_equals(mtarget, ixork ^ tm)) {
            return add_xor_3_gate(st, gi, gk, gm, opt);
          }
          if (ttable_equals(mtarget, iandm ^ tk)) {
            return add_and_xor_gate(st, gi, gm, gk, opt);
          }
          if (ttable_equals(mtarget, kandm ^ ti)) {
            return add_and_xor_gate(st, gk, gm, gi, opt);
          }
          ttable ixorm = ti ^ tm;
          if (ttable_equals(mtarget, ixorm | tk)) {
            return add_xor_or_gate(st, gi, gm, gk, opt);
          }
          if (ttable_equals(mtarget, ixorm & tk)) {
            return add_xor_and_gate(st, gi, gm, gk, opt);
          }
          ttable kxorm = tk ^ tm;
          if (ttable_equals(mtarget, kxorm | ti)) {
            return add_xor_or_gate(st, gk, gm, gi, opt);
          }
          if (ttable_equals(mtarget, kxorm & ti)) {
            return add_xor_and_gate(st, gk, gm, gi, opt);
          }
          if (ttable_equals(mtarget, iorm ^ tk)) {
            return add_or_xor_gate(st, gi, gm, gk, opt);
          }
          if (ttable_equals(mtarget, korm ^ ti)) {
            return add_or_xor_gate(st, gk, gm, gi, opt);
          }
        }
      }
    }
  }

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
    if (opt->lut_graph) {
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
    } else {
      state nst_and = *st; /* New state using AND multiplexer. */

      /* A multiplexer will have to be added later. */
      nst_and.max_gates -= 2;
      nst_and.max_sat_metric -= get_sat_metric(AND) + get_sat_metric(XOR);

      gatenum fb = create_circuit(&nst_and, target & ~fsel, mask & ~fsel, next_inbits, opt);
      gatenum mux_out_and = NO_GATE;
      if (fb != NO_GATE) {
        gatenum fc = create_circuit(&nst_and, nst_and.gates[fb].table ^ target, mask & fsel,
            next_inbits, opt);
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
      gatenum mux_out_or = NO_GATE;
      if (fd != NO_GATE) {
        gatenum fe = create_circuit(&nst_or, nst_or.gates[fd].table ^ target, mask & ~fsel,
            next_inbits, opt);
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
    }

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
  }

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

    if (work.st.num_gates >= 5 && search_5lut(work.st, work.target, work.mask, work.inbits, res)) {
      continue;
    }
    bool search7;
    MPI_Bcast(&search7, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (search7 && work.st.num_gates >= 7) {
      search_7lut(work.st, work.target, work.mask, work.inbits, res);
    }
  }
}

/* If sbox is true, a target truth table for the given bit of the sbox is generated.
   If sbox is false, the truth table of the given input bit is generated. */
ttable generate_target(uint8_t bit, bool sbox) {
  assert(bit < 8);
  uint64_t vec[] = {0, 0, 0, 0};
  uint64_t *var = &vec[0];
  for (uint16_t i = 0; i < 256; i++) {
    if (i == 64) {
      var = &vec[1];
    } else if (i == 128) {
      var = &vec[2];
    } else if (i == 192) {
      var = &vec[3];
    }
    *var >>= 1;
    *var |= (uint64_t)(((sbox ? g_sbox_enc[i] : i) >> bit) & 1) << 63;
  }
  ttable t;
  memcpy(&t, &vec, sizeof(ttable));
  return t;
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
  int work_block_lengths[] = {1, 4, 4, 8, 1};
  MPI_Aint work_displacements[] = {
      offsetof(mpi_work, st),
      offsetof(mpi_work, target),
      offsetof(mpi_work, mask),
      offsetof(mpi_work, inbits),
      offsetof(mpi_work, quit)
    };
  MPI_Datatype work_datatypes[] = {
      state_type,
      MPI_UINT64_T,
      MPI_UINT64_T,
      MPI_UINT8_T,
      MPI_C_BOOL
    };
  assert(MPI_Type_create_struct(5, work_block_lengths, work_displacements, work_datatypes,
      &g_mpi_work_type) == MPI_SUCCESS);
  assert(MPI_Type_commit(&g_mpi_work_type) == MPI_SUCCESS);
}

/* Print the program usage help to stdout. */
void print_command_help(const char *name) {
  assert(name != NULL);
  printf(
            "\nsboxgates generates graphs of Boolean gates or 3-input LUTs that realize a\n"
            "specified S-box. The program uses MPI for parallellization and should therefore\n"
            "be run using the mpirun utility. Generated graphs are output as XML files.\n"
            "In its basic mode, the program generates a single graph for all outputs of the\n"
            "S-box. It is also possible to generate separate graphs for each output, which\n"
            "can significantly decrease the time to generate the graph.\n\n"

            "Generated graphs can be converted to C/CUDA source code or to Graphviz DOT\n"
            "format.\n\n"

            "Arguments:\n"
            "-b file   Target S-box definition file. (Mandatory)\n"
            "-g file   Load graph from file as initial state. (For use with -o.)\n"
            "-h        Display this help.\n"
            "-i n      Do n iterations per step.\n"
            "-l        Generate LUT graph. Results in smaller graphs but takes significantly\n"
            "          longer time.\n"
            "-n        Use ANDNOT gates in addition to NOT, AND, OR, and XOR.\n"
            "-o n      Generate one-output graph for output n.\n"
            "-p value  Permute sbox by XORing input with value.\n"
            "-s        Use SAT metric to optimize the generated graph for use with SAT\n"
            "          solvers.\n\n"

            "Graph conversion arguments, used alone:\n"
            "-c file   Convert file to C/CUDA function.\n"
            "-d file   Convert file to DOT digraph.\n\n"

            "Examples:\n"
            "mpirun %s -b sboxes/des_s1.txt\n"
            "Generates a gate graph for all output bits of the DES S1 S-box.\n\n"

            "mpirun %s -l -b sboxes/des_s1.txt\n"
            "Generates a LUT graph for all output bits of the DES S1 S-box.\n\n"

            "mpirun %s -o 0 -b sboxes/rijndael.txt\n"
            "Generates a gate graph for output bit 0 of the Rijndael S-box.\n\n"

            "mpirun %s -d 2-017-0000-01-b55885b4.xml | dot -Tpng > 2-017-0000-01-b55885b4.png\n"
            "Converts a generated graph to Graphviz DOT format and generates a graphical\n"
            "representation.\n\n", name, name, name, name);
}

/* Used in parse_options to increase readability. */
#define PARSE_OPTIONS_TEST_NAME_LENGTH(X)\
  if (strlen(X) >= MAX_NAME_LEN) {\
    fprintf(stderr, "Error: File name too long. (sboxgates.c:%d)\n", __LINE__);\
    return false;\
  }

/* Parses command line options and places the result in an options structure. In case of error,
   the function will return false and set retval to a suggested return value for main. */
bool parse_options(int argc, char **argv, options *opt, int *retval) {
  assert(argc > 0);
  assert(argv != NULL);
  assert(opt != NULL);
  assert(retval != NULL);
  char *name = argv[0];

  if (argc == 1) {
    print_command_help(name);
    *retval = 1;
    return false;
  }

  memset(opt, 0, sizeof(options));
  opt->oneoutput = -1;
  opt->iterations = 1;
  opt->randomize = true;
  opt->metric = GATES;

  char opts[] = "b:c:d:g:hi:lno:p:s";

  int c;
  char *endptr;
  while ((c = getopt(argc, argv, opts)) != -1) {
    switch (c) {
      /* Set target S-box definition. */
      case 'b':
        PARSE_OPTIONS_TEST_NAME_LENGTH(optarg);
        strcpy(opt->sboxfname, optarg);
        break;
      /* Convert generated graph to C function. */
      case 'c':
        opt->output_c = true;
        PARSE_OPTIONS_TEST_NAME_LENGTH(optarg);
        strcpy(opt->fname, optarg);
        break;
      /* Convert generated graph to DOT digraph. */
      case 'd':
        opt->output_dot = true;
        PARSE_OPTIONS_TEST_NAME_LENGTH(optarg);
        strcpy(opt->fname, optarg);
        break;
      /* Load graph from file. */
      case 'g':
        PARSE_OPTIONS_TEST_NAME_LENGTH(optarg);
        strcpy(opt->gfname, optarg);
        break;
      /* Print help. */
      case 'h':
        print_command_help(name);
        *retval = 0;
        return false;
      /* Do multiple iterations per step. */
      case 'i':
        opt->iterations = strtoul(optarg, &endptr, 10);
        if (*endptr != '\0' || opt->iterations < 1) {
          fprintf(stderr, "Bad iterations value: %s (sboxgates.c:%d)\n", optarg, __LINE__);
        }
        break;
      /* Generate 3LUT graph. */
      case 'l':
        opt->lut_graph = true;
        break;
      /* Use ANDNOT gates. */
      case 'n':
        opt->andnot = true;
        break;
      /* Generate single-output graph. */
      case 'o':
        opt->oneoutput = strtoul(optarg, &endptr, 10);
        if (*endptr != '\0' || opt->oneoutput < 0 || opt->oneoutput > 7) {
          fprintf(stderr, "Bad output value: %s (sboxgates.c:%d)\n", optarg, __LINE__);
          *retval = 1;
          return false;
        }
        break;
      /* Permute S-box by XORing input with value. */
      case 'p':
        opt->permute = strtoul(optarg, &endptr, 10);
        if (*endptr != '\0' || opt->permute < 0 || opt->permute > 255) {
          fprintf(stderr, "Bad permutation value: %s (sboxgates.c:%d)\n", optarg, __LINE__);
          *retval = 1;
          return false;
        }
        break;
      /* Use SAT metric. */
      case 's':
        opt->metric = SAT;
        break;
      /* Undefined flag. */
      default:
        fprintf(stderr, "Bad argument. (sboxgates.c:%d)\n", __LINE__);
        *retval = 1;
        return false;
    }
  }

  if (opt->output_c && opt->output_dot) {
    fprintf(stderr, "Cannot combine c and d options. (sboxgates.c:%d)\n", __LINE__);
    *retval = 1;
    return false;
  }

  if (opt->lut_graph && opt->metric == SAT) {
    fprintf(stderr, "SAT metric can not be combined with LUT graph generation. (sboxgates.c:%d)\n",
        __LINE__);
    *retval = 1;
    return false;
  }

  if (!opt->output_c && !opt->output_dot && strlen(opt->sboxfname) == 0) {
    fprintf(stderr, "No target S-box file name argument. (sboxgates.c:%d)\n", __LINE__);
    *retval = 1;
    return false;
  }

  return true;
}

/* Loads an S-box from a file. The file should contain the S-box table as 2^n (1 <= n <= 8)
   whitespace separated hexadecimal numbers. The S-box is loaded into the 256 item array pointed to
   by sbox and num_input is set to the calculated number of input bits. The input file name is
   taken from the opt structure. */
bool load_sbox(uint8_t *sbox, uint32_t *num_inputs, const options *opt) {
  assert(sbox != NULL);
  assert(num_inputs != NULL);
  assert(opt != NULL);
  assert(opt->sboxfname != NULL);
  int sbox_inp = 0;

  FILE *fp = fopen(opt->sboxfname, "r");
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
    for (int i = 0; i < 256; i++) {
      sbox[i] = target_sbox[i ^ (uint8_t)opt->permute];
    }
  }
  return true;
}

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
  options opt;
  int retval;
  if (!parse_options(argc, argv, &opt, &retval)) {
    stop_workers();
    MPI_Finalize();
    return retval;
  }

  /* Convert graph to C or DOT output and quit. */
  if (opt.output_c || opt.output_dot) {
    stop_workers();
    state st;
    if (!load_state(opt.fname, &st)) {
      fprintf(stderr, "Error when reading state file. (sboxgates.c:%d)\n", __LINE__);
      return false;
    }
    int retval = 0;
    if (opt.output_c) {
      if (!print_c_function(st)) {
        retval = 1;
      }
    } else {
      print_digraph(st);
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
