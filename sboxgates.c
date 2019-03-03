/* sboxgates.c

   Program for finding low gate count implementations of S-boxes.
   The algorithm used is described in Kwan, Matthew: "Reducing the Gate Count of Bitslice DES."
   IACR Cryptology ePrint Archive 2000 (2000): 51. Improvements from
   SBOXDiscovery (https://github.com/DeepLearningJohnDoe/SBOXDiscovery) have been added.

   Copyright (c) 2016-2017, 2019 Marcus Dansarie

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
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <x86intrin.h>
#include "convert_graph.h"
#include "lut.h"
#include "sboxgates.h"
#include "state.h"

typedef struct {
  state st;
  ttable target;
  ttable mask;
  bool quit;
} mpi_work;

uint8_t g_sbox_enc[256];    /* Target S-box. */

ttable g_target[8];       /* Truth tables for the output bits of the sbox. */
metric g_metric = GATES;  /* Metric that should be used when selecting between two solutions. */

/* Test two truth tables for equality. */
static inline bool ttable_equals(const ttable in1, const ttable in2) {
  ttable res = in1 ^ in2;
  return _mm256_testz_si256(res, res);
}

/* Performs a masked test for equality. Only bits set to 1 in the mask will be tested. */
bool ttable_equals_mask(const ttable in1, const ttable in2, const ttable mask) {
  ttable res = (in1 ^ in2) & mask;
  return _mm256_testz_si256(res, res);
}

/* Adds a gate to the state st. Returns the gate id of the added gate. If an input gate is
   equal to NO_GATE (only gid1 in case of a NOT gate), NO_GATE will be returned. */
static inline gatenum add_gate(state *st, gate_type type, ttable table, gatenum gid1,
    gatenum gid2) {
  if (gid1 == NO_GATE || (gid2 == NO_GATE && type != NOT) || st->num_gates > st->max_gates) {
    return NO_GATE;
  }
  if (g_metric == SAT && st->sat_metric > st->max_sat_metric) {
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
static inline gatenum add_lut(state *st, uint8_t func, ttable table, gatenum gid1, gatenum gid2,
    gatenum gid3) {
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

static inline gatenum add_not_gate(state *st, gatenum gid) {
  if (gid == NO_GATE) {
    return NO_GATE;
  }
  return add_gate(st, NOT, ~st->gates[gid].table, gid, NO_GATE);
}

static inline gatenum add_and_gate(state *st, gatenum gid1, gatenum gid2) {
  if (gid1 == NO_GATE || gid2 == NO_GATE) {
    return NO_GATE;
  }
  if (gid1 == gid2) {
    return gid1;
  }
  return add_gate(st, AND, st->gates[gid1].table & st->gates[gid2].table, gid1, gid2);
}

static inline gatenum add_or_gate(state *st, gatenum gid1, gatenum gid2) {
  if (gid1 == NO_GATE || gid2 == NO_GATE) {
    return NO_GATE;
  }
  if (gid1 == gid2) {
    return gid1;
  }
  return add_gate(st, OR, st->gates[gid1].table | st->gates[gid2].table, gid1, gid2);
}

static inline gatenum add_xor_gate(state *st, gatenum gid1, gatenum gid2) {
  if (gid1 == NO_GATE || gid2 == NO_GATE) {
    return NO_GATE;
  }
  return add_gate(st, XOR, st->gates[gid1].table ^ st->gates[gid2].table, gid1, gid2);
}

static inline gatenum add_andnot_gate(state *st, gatenum gid1, gatenum gid2) {
  if (gid1 == NO_GATE || gid2 == NO_GATE) {
    return NO_GATE;
  }
  return add_gate(st, ANDNOT, ~st->gates[gid1].table & st->gates[gid2].table, gid1, gid2);
}

static inline gatenum add_nand_gate(state *st, gatenum gid1, gatenum gid2) {
  return add_not_gate(st, add_and_gate(st, gid1, gid2));
}

static inline gatenum add_nor_gate(state *st, gatenum gid1, gatenum gid2) {
  return add_not_gate(st, add_or_gate(st, gid1, gid2));
}

static inline gatenum add_xnor_gate(state *st, gatenum gid1, gatenum gid2) {
  return add_not_gate(st, add_xor_gate(st, gid1, gid2));
}

static inline gatenum add_or_not_gate(state *st, gatenum gid1, gatenum gid2) {
  return add_or_gate(st, add_not_gate(st, gid1), gid2);
}

static inline gatenum add_and_not_gate(state *st, gatenum gid1, gatenum gid2) {
  return add_and_gate(st, add_not_gate(st, gid1), gid2);
}

static inline gatenum add_or_3_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3) {
  return add_or_gate(st, add_or_gate(st, gid1, gid2), gid3);
}

static inline gatenum add_and_3_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3) {
  return add_and_gate(st, add_and_gate(st, gid1, gid2), gid3);
}

static inline gatenum add_xor_3_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3) {
  return add_xor_gate(st, add_xor_gate(st, gid1, gid2), gid3);
}

static inline gatenum add_and_or_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3) {
  return add_or_gate(st, add_and_gate(st, gid1, gid2), gid3);
}

static inline gatenum add_and_xor_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3) {
  return add_xor_gate(st, add_and_gate(st, gid1, gid2), gid3);
}

static inline gatenum add_xor_or_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3) {
  return add_or_gate(st, add_xor_gate(st, gid1, gid2), gid3);
}

static inline gatenum add_xor_and_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3) {
  return add_and_gate(st, add_xor_gate(st, gid1, gid2), gid3);
}

static inline gatenum add_or_and_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3) {
  return add_and_gate(st, add_or_gate(st, gid1, gid2), gid3);
}

static inline gatenum add_or_xor_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3) {
  return add_xor_gate(st, add_or_gate(st, gid1, gid2), gid3);
}

static inline gatenum add_andnot_or_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3) {
  return add_or_gate(st, add_andnot_gate(st, gid1, gid2), gid3);
}

static inline gatenum add_xor_andnot_a_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3) {
  return add_andnot_gate(st, gid3, add_xor_gate(st, gid1, gid2));
}

static inline gatenum add_xor_andnot_b_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3) {
  return add_andnot_gate(st, add_xor_gate(st, gid1, gid2), gid3);
}

static inline gatenum add_and_andnot_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3) {
  return add_and_gate(st, add_andnot_gate(st, gid1, gid2), gid3);
}

static inline gatenum add_andnot_3_a_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3) {
  return add_andnot_gate(st, gid1, add_andnot_gate(st, gid2, gid3));
}

static inline gatenum add_andnot_3_b_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3) {
  return add_andnot_gate(st, add_andnot_gate(st, gid1, gid2), gid3);
}

static inline gatenum add_andnot_xor_gate(state *st, gatenum gid1, gatenum gid2, gatenum gid3) {
  return add_xor_gate(st, add_andnot_gate(st, gid1, gid2), gid3);
}

/* Generates pseudorandom 64 bit strings. Used for randomizing the search process. */
uint64_t xorshift1024() {
  static bool init = false;
  static uint64_t rand[16];
  static int p = 0;
  if (!init) {
    FILE *rand_fp = fopen("/dev/urandom", "r");
    if (rand_fp == NULL) {
      fprintf(stderr, "Error opening /dev/urandom.\n");
    } else if (fread(rand, 16 * sizeof(uint64_t), 1, rand_fp) != 1) {
      fprintf(stderr, "Error reading from /dev/urandom.\n");
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

/* Recursively builds the gate network. The numbered comments are references to Matthew Kwan's
   paper. */
static gatenum create_circuit(state *st, const ttable target, const ttable mask,
    const int8_t *inbits, const bool andnot, const bool lut, const bool randomize) {

  gatenum gate_order[MAX_GATES];
  for (int i = 0; i < st->num_gates; i++) {
    gate_order[i] = st->num_gates - 1 - i;
  }

  if (randomize) {
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

  for (int i = 0; i < st->num_gates; i++) {
    if (ttable_equals_mask(target, ~st->gates[gate_order[i]].table, mask)) {
      return add_not_gate(st, gate_order[i]);
    }
  }

  /* 3. Look at all pairs of gates in the existing circuit. If they can be combined with a single
     gate to produce the desired map, add that single gate and return its ID. */

  const ttable mtarget = target & mask;
  for (int i = 0; i < st->num_gates; i++) {
    const gatenum gi = gate_order[i];
    const ttable ti = st->gates[gi].table & mask;
    for (int k = i + 1; k < st->num_gates; k++) {
      const gatenum gk = gate_order[k];
      const ttable tk = st->gates[gk].table & mask;
      if (ttable_equals(mtarget, ti | tk)) {
        return add_or_gate(st, gi, gk);
      }
      if (ttable_equals(mtarget, ti & tk)) {
        return add_and_gate(st, gi, gk);
      }
      if (andnot) {
        if (ttable_equals_mask(target, ~ti & tk, mask)) {
          return add_andnot_gate(st, gi, gk);
        }
        if (ttable_equals_mask(target, ~tk & ti, mask)) {
          return add_andnot_gate(st, gk, gi);
        }
      }
      if (ttable_equals(mtarget, ti ^ tk)) {
        return add_xor_gate(st, gi, gk);
      }
    }
  }

  if (lut) {
    /* Look through all combinations of three gates in the circuit. For each combination, check if
       any of the 256 possible three bit Boolean functions produces the desired map. If so, add that
       LUT and return the ID. */
    for (int i = 0; i < st->num_gates; i++) {
      const gatenum gi = gate_order[i];
      const ttable ta = st->gates[gi].table;
      for (int k = i + 1; k < st->num_gates; k++) {
        const gatenum gk = gate_order[k];
        const ttable tb = st->gates[gk].table;
        for (int m = k + 1; m < st->num_gates; m++) {
          const gatenum gm = gate_order[m];
          const ttable tc = st->gates[gm].table;
          if (!check_3lut_possible(target, mask, ta, tb, tc)) {
            continue;
          }
          uint8_t func;
          if (!get_lut_function(ta, tb, tc, target, mask, randomize, &func)) {
            continue;
          }
          ttable nt = generate_lut_ttable(func, ta, tb, tc);
          assert(ttable_equals_mask(target, nt, mask));
          return add_lut(st, func, nt, gi, gk, gm);
        }
      }
    }

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Broadcast work to be done. */
    mpi_work work = {*st, target, mask, false};
    MPI_Bcast(&work, sizeof(work), MPI_BYTE, 0, MPI_COMM_WORLD);

    /* Look through all combinations of five gates in the circuit. For each combination, check if
       a combination of two of the possible 256 three bit Boolean functions as in
       LUT(LUT(a,b,c),d,e) produces the desired map. If so, add those LUTs and return the ID of the
       output LUT. */

    uint16_t res[10];

    memset(res, 0, sizeof(uint16_t) * 10);
    printf("[   0] Search 5.\n");

    if (search_5lut(work.st, work.target, work.mask, res)) {
      uint8_t func_outer = (uint8_t)res[0];
      uint8_t func_inner = (uint8_t)res[1];
      gatenum a = res[2];
      gatenum b = res[3];
      gatenum c = res[4];
      gatenum d = res[5];
      gatenum e = res[6];
      ttable ta = st->gates[a].table;
      ttable tb = st->gates[b].table;
      ttable tc = st->gates[c].table;
      ttable td = st->gates[d].table;
      ttable te = st->gates[e].table;
      printf("[   0] Found 5LUT: %02x %02x    %3d %3d %3d %3d %3d\n",
          func_outer, func_inner, a, b, c, d, e);

      assert(check_5lut_possible(target, mask, ta, tb, tc, td, te));
      ttable t_outer = generate_lut_ttable(func_outer, ta, tb, tc);
      ttable t_inner = generate_lut_ttable(func_inner, t_outer, td, te);
      assert(ttable_equals_mask(target, t_inner, mask));

      return add_lut(st, func_inner, t_inner,
          add_lut(st, func_outer, t_outer, a, b, c), d, e);
    }

    printf("[   0] Search 7.\n");
    if (search_7lut(work.st, work.target, work.mask, res)) {
      uint8_t func_outer = (uint8_t)res[0];
      uint8_t func_middle = (uint8_t)res[1];
      uint8_t func_inner = (uint8_t)res[2];
      gatenum a = res[3];
      gatenum b = res[4];
      gatenum c = res[5];
      gatenum d = res[6];
      gatenum e = res[7];
      gatenum f = res[8];
      gatenum g = res[9];
      ttable ta = st->gates[a].table;
      ttable tb = st->gates[b].table;
      ttable tc = st->gates[c].table;
      ttable td = st->gates[d].table;
      ttable te = st->gates[e].table;
      ttable tf = st->gates[f].table;
      ttable tg = st->gates[g].table;
      printf("[   0] Found 7LUT: %02x %02x %02x %3d %3d %3d %3d %3d %3d %3d\n",
          func_outer, func_middle, func_inner, a, b, c, d, e, f, g);
      assert(check_7lut_possible(target, mask, ta, tb, tc, td, te, tf, tg));
      ttable t_outer = generate_lut_ttable(func_outer, ta, tb, tc);
      ttable t_middle = generate_lut_ttable(func_middle, td, te, tf);
      ttable t_inner = generate_lut_ttable(func_inner, t_outer, t_middle, tg);
      assert(ttable_equals_mask(target, t_inner, mask));
      return add_lut(st, func_inner, t_inner,
          add_lut(st, func_outer, t_outer, a, b, c),
          add_lut(st, func_middle, t_middle, d, e, f), g);
    }

    printf("[   0] No LUTs found. Num gates: %d\n", st->num_gates - 8);
  } else {
    /* 4. Look at all combinations of two or three gates in the circuit. If they can be combined
       with two gates to produce the desired map, add the gates, and return the ID of the one that
       produces the desired map. */

    for (int i = 0; i < st->num_gates; i++) {
      const gatenum gi = gate_order[i];
      ttable ti = st->gates[gi].table;
      for (int k = i + 1; k < st->num_gates; k++) {
        const gatenum gk = gate_order[k];
        ttable tk = st->gates[gk].table;
        if (ttable_equals_mask(target, ~(ti | tk), mask)) {
          return add_nor_gate(st, gi, gk);
        }
        if (ttable_equals_mask(target, ~(ti & tk), mask)) {
          return add_nand_gate(st, gi, gk);
        }
        if (ttable_equals_mask(target, ~ti | tk, mask)) {
          return add_or_not_gate(st, gi, gk);
        }
        if (ttable_equals_mask(target, ~tk | ti, mask)) {
          return add_or_not_gate(st, gk, gi);
        }
        if (!andnot) {
          if (ttable_equals_mask(target, ~ti & tk, mask)) {
            return add_and_not_gate(st, gi, gk);
          }
          if (ttable_equals_mask(target, ~tk & ti, mask)) {
            return add_and_not_gate(st, gk, gi);
          }
        } else if (ttable_equals_mask(target, ~ti & ~tk, mask)) {
          return add_andnot_gate(st, gi, add_not_gate(st, gk));
        }
        if (ttable_equals_mask(target, ~(ti ^ tk), mask)) {
          return add_xnor_gate(st, gi, gk);
        }
      }
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
          if (!check_3lut_possible(target, mask, ti, tk, tm)) {
            continue;
          }
          if (ttable_equals(mtarget, iandk & tm)) {
            return add_and_3_gate(st, gi, gk, gm);
          }
          if (ttable_equals(mtarget, iandk | tm)) {
            return add_and_or_gate(st, gi, gk, gm);
          }
          if (ttable_equals(mtarget, iork | tm)) {
            return add_or_3_gate(st, gi, gk, gm);
          }
          if (ttable_equals(mtarget, iork & tm)) {
            return add_or_and_gate(st, gi, gk, gm);
          }
          ttable iandm = ti & tm;
          if (ttable_equals(mtarget, iandm | tk)) {
            return add_and_or_gate(st, gi, gm, gk);
          }
          ttable kandm = tk & tm;
          if (ttable_equals(mtarget, kandm | ti)) {
            return add_and_or_gate(st, gk, gm, gi);
          }
          ttable iorm = ti | tm;
          if (ttable_equals(mtarget, iorm & tk)) {
            return add_or_and_gate(st, gi, gm, gk);
          }
          ttable korm = tk | tm;
          if (ttable_equals(mtarget, korm & ti)) {
            return add_or_and_gate(st, gk, gm, gi);
          }
          if (andnot) {
            if (ttable_equals(mtarget, ti | (~tk & tm))) {
              return add_andnot_or_gate(st, gk, gm, gi);
            }
            if (ttable_equals(mtarget, ti | (tk & ~tm))) {
              return add_andnot_or_gate(st, gm, gk, gi);
            }
            if (ttable_equals(mtarget, tm | (~ti & tk))) {
              return add_andnot_or_gate(st, gi, gk, gm);
            }
            if (ttable_equals(mtarget, tm | (ti & ~tk))) {
              return add_andnot_or_gate(st, gk, gi, gm);
            }
            if (ttable_equals(mtarget, tk | (~ti & tm))) {
              return add_andnot_or_gate(st, gi, gm, gk);
            }
            if (ttable_equals(mtarget, tk | (ti & ~tm))) {
              return add_andnot_or_gate(st, gm, gi, gk);
            }
            if (ttable_equals(mtarget, ~ti & tk & tm)) {
              return add_and_andnot_gate(st, gi, gk, gm);
            }
            if (ttable_equals(mtarget, ti & ~tk & tm)) {
              return add_and_andnot_gate(st, gk, gi, gm);
            }
            if (ttable_equals(mtarget, ti & tk & ~tm)) {
              return add_and_andnot_gate(st, gm, gk, gi);
            }
            if (ttable_equals(mtarget, ~ti & ~tk & tm)) {
              return add_andnot_3_a_gate(st, gi, gk, gm);
            }
            if (ttable_equals(mtarget, ~ti & tk & ~tm)) {
              return add_andnot_3_a_gate(st, gi, gm, gk);
            }
            if (ttable_equals(mtarget, ti & ~tk & ~tm)) {
              return add_andnot_3_a_gate(st, gk, gm, gi);
            }
            if (ttable_equals(mtarget, ti & ~(~tk & tm))) {
              return add_andnot_3_b_gate(st, gk, gm, gi);
            }
            if (ttable_equals(mtarget, ti & ~(tk & ~tm))) {
              return add_andnot_3_b_gate(st, gm, gk, gi);
            }
            if (ttable_equals(mtarget, tk & ~(~ti & tm))) {
              return add_andnot_3_b_gate(st, gi, gm, gk);
            }
            if (ttable_equals(mtarget, tk & ~(ti & ~tm))) {
              return add_andnot_3_b_gate(st, gm, gi, gk);
            }
            if (ttable_equals(mtarget, tm & ~(~tk & ti))) {
              return add_andnot_3_b_gate(st, gk, gi, gm);
            }
            if (ttable_equals(mtarget, tm & ~(tk & ~ti))) {
              return add_andnot_3_b_gate(st, gi, gk, gm);
            }
            if (ttable_equals(mtarget, ~ti & (tk ^ tm))) {
              return add_xor_andnot_a_gate(st, gk, gm, gi);
            }
            if (ttable_equals(mtarget, ~tk & (ti ^ tm))) {
              return add_xor_andnot_a_gate(st, gi, gm, gk);
            }
            if (ttable_equals(mtarget, ~tm & (tk ^ ti))) {
              return add_xor_andnot_a_gate(st, gk, gi, gm);
            }
            if (ttable_equals(mtarget, ti & ~(tk ^ tm))) {
              return add_xor_andnot_b_gate(st, gk, gm, gi);
            }
            if (ttable_equals(mtarget, tk & ~(ti ^ tm))) {
              return add_xor_andnot_b_gate(st, gi, gm, gk);
            }
            if (ttable_equals(mtarget, tm & ~(tk ^ ti))) {
              return add_xor_andnot_b_gate(st, gk, gi, gm);
            }
            if (ttable_equals(mtarget, ti ^ (~tk & tm))) {
              return add_andnot_xor_gate(st, gk, gm, gi);
            }
            if (ttable_equals(mtarget, ti ^ (tk & ~tm))) {
              return add_andnot_xor_gate(st, gm, gk, gi);
            }
            if (ttable_equals(mtarget, tk ^ (~ti & tm))) {
              return add_andnot_xor_gate(st, gi, gm, gk);
            }
            if (ttable_equals(mtarget, tk ^ (ti & ~tm))) {
              return add_andnot_xor_gate(st, gm, gi, gk);
            }
            if (ttable_equals(mtarget, tm ^ (~tk & ti))) {
              return add_andnot_xor_gate(st, gk, gi, gm);
            }
            if (ttable_equals(mtarget, tm ^ (tk & ~ti))) {
              return add_andnot_xor_gate(st, gi, gk, gm);
            }
          }
          if (ttable_equals(mtarget, ixork | tm)) {
            return add_xor_or_gate(st, gi, gk, gm);
          }
          if (ttable_equals(mtarget, ixork & tm)) {
            return add_xor_and_gate(st, gi, gk, gm);
          }
          if (ttable_equals(mtarget, iandk ^ tm)) {
            return add_and_xor_gate(st, gi, gk, gm);
          }
          if (ttable_equals(mtarget, iork ^ tm)) {
            return add_or_xor_gate(st, gi, gk, gm);
          }
          if (ttable_equals(mtarget, ixork ^ tm)) {
            return add_xor_3_gate(st, gi, gk, gm);
          }
          if (ttable_equals(mtarget, iandm ^ tk)) {
            return add_and_xor_gate(st, gi, gm, gk);
          }
          if (ttable_equals(mtarget, kandm ^ ti)) {
            return add_and_xor_gate(st, gk, gm, gi);
          }
          ttable ixorm = ti ^ tm;
          if (ttable_equals(mtarget, ixorm | tk)) {
            return add_xor_or_gate(st, gi, gm, gk);
          }
          if (ttable_equals(mtarget, ixorm & tk)) {
            return add_xor_and_gate(st, gi, gm, gk);
          }
          ttable kxorm = tk ^ tm;
          if (ttable_equals(mtarget, kxorm | ti)) {
            return add_xor_or_gate(st, gk, gm, gi);
          }
          if (ttable_equals(mtarget, kxorm & ti)) {
            return add_xor_and_gate(st, gk, gm, gi);
          }
          if (ttable_equals(mtarget, iorm ^ tk)) {
            return add_or_xor_gate(st, gi, gm, gk);
          }
          if (ttable_equals(mtarget, korm ^ ti)) {
            return add_or_xor_gate(st, gk, gm, gi);
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
  for (int bit = 0; bit < 8; bit++) {
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
    if (lut) {
      nst = *st;
      gatenum fb = create_circuit(&nst, target, mask & ~fsel, next_inbits, andnot, true, randomize);
      if (fb == NO_GATE) {
        continue;
      }
      assert(ttable_equals_mask(target, nst.gates[fb].table, mask & ~fsel));
      gatenum fc = create_circuit(&nst, target, mask & fsel, next_inbits, andnot, true, randomize);
      if (fc == NO_GATE) {
        continue;
      }
      assert(ttable_equals_mask(target, nst.gates[fc].table, mask & fsel));

      if (fb == fc) {
        nst_out = fb;
        assert(ttable_equals_mask(target, nst.gates[nst_out].table, mask));
      } else if (fb == bit) {
        nst_out = add_and_gate(&nst, fb, fc);
        if (nst_out == NO_GATE) {
          continue;
        }
        assert(ttable_equals_mask(target, nst.gates[nst_out].table, mask));
      } else if (fc == bit) {
        nst_out = add_or_gate(&nst, fb, fc);
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
      gatenum fb = create_circuit(&nst_and, target & ~fsel, mask & ~fsel, next_inbits, andnot,
          false, randomize);
      gatenum mux_out_and = NO_GATE;
      if (fb != NO_GATE) {
        gatenum fc = create_circuit(&nst_and, nst_and.gates[fb].table ^ target, mask & fsel,
            next_inbits, andnot, false, randomize);
        gatenum andg = add_and_gate(&nst_and, fc, bit);
        mux_out_and = add_xor_gate(&nst_and, fb, andg);
        assert(mux_out_and == NO_GATE ||
            ttable_equals_mask(target, nst_and.gates[mux_out_and].table, mask));
      }

      state nst_or = *st; /* New state using OR multiplexer. */
      if (mux_out_and != NO_GATE) {
        nst_or.max_gates = nst_and.num_gates;
        nst_or.max_sat_metric = nst_and.sat_metric;
      }
      gatenum fd = create_circuit(&nst_or, ~target & fsel, mask & fsel, next_inbits, andnot, false,
          randomize);
      gatenum mux_out_or = NO_GATE;
      if (fd != NO_GATE) {
        gatenum fe = create_circuit(&nst_or, nst_or.gates[fd].table ^ target, mask & ~fsel,
            next_inbits, andnot, false, randomize);
        gatenum org = add_or_gate(&nst_or, fe, bit);
        mux_out_or = add_xor_gate(&nst_or, fd, org);
        assert(mux_out_or == NO_GATE ||
            ttable_equals_mask(target, nst_or.gates[mux_out_or].table, mask));
        nst_or.max_gates = st->max_gates;
        nst_or.max_sat_metric = st->max_sat_metric;
      }
      if (mux_out_and == NO_GATE && mux_out_or == NO_GATE) {
        continue;
      }

      if (g_metric == GATES) {
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
    if (g_metric == GATES) {
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
    MPI_Bcast(&work, sizeof(work), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (work.quit) {
      return;
    }

    if (search_5lut(work.st, work.target, work.mask, res)) {
      continue;
    }
    search_7lut(work.st, work.target, work.mask, res);
  }
}

/* If sbox is true, a target truth table for the given bit of the sbox is generated.
   If sbox is false, the truth table of the given input bit is generated. */
static ttable generate_target(uint8_t bit, bool sbox) {
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
  return _mm256_loadu_si256((ttable*)vec);
}

void generate_graph_one_output(const bool andnot, const bool lut, const bool randomize,
    const int iterations, const int output, state st) {
  assert(iterations > 0);
  assert(output >= 0 && output <= 7);
  printf("Generating graphs for output %d...\n", output);
  for (int iter = 0; iter < iterations; iter++) {
    state nst = st;

    int8_t bits[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    const ttable mask = ~_mm256_setzero_si256();
    nst.outputs[output] = create_circuit(&nst, g_target[output], mask, bits, andnot, lut,
        randomize);
    if (nst.outputs[output] == NO_GATE) {
      printf("(%d/%d): Not found.\n", iter + 1, iterations);
      continue;
    }
    printf("(%d/%d): %d gates. SAT metric: %d\n", iter + 1, iterations, nst.num_gates - 8,
        nst.sat_metric);
    save_state(nst);
    if (g_metric == GATES) {
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
void generate_graph(const bool andnot, const bool lut, const bool randomize, const int iterations,
    const state st) {
  int num_start_states = 1;
  state start_states[20];
  start_states[0] = st;

  /* Build the gate network one output at a time. After every added output, select the gate network
     or network with the least amount of gates and add another. */
  int num_outputs;
  while ((num_outputs = count_state_outputs(start_states[0])) < 8) {
    gatenum max_gates = MAX_GATES;
    int max_sat_metric = INT_MAX;
    state out_states[20];
    int num_out_states = 0;

    for (int iter = 0; iter < iterations; iter++) {
      printf("Generating circuits with %d output%s. (%d/%d)\n", num_outputs + 1,
          num_outputs == 0 ? "" : "s", iter + 1, iterations);
      for (uint8_t current_state = 0; current_state < num_start_states; current_state++) {
        start_states[current_state].max_gates = max_gates;
        start_states[current_state].max_sat_metric = max_sat_metric;

        /* Add all outputs not already present to see which resulting network is the smallest. */
        for (uint8_t output = 0; output < 8; output++) {
          if (start_states[current_state].outputs[output] != NO_GATE) {
            printf("Skipping output %d.\n", output);
            continue;
          }
          printf("Generating circuit for output %d...\n", output);
          int8_t bits[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
          state st = start_states[current_state];
          if (g_metric == GATES) {
            st.max_gates = max_gates;
          } else {
            st.max_sat_metric = max_sat_metric;
          }
          const ttable mask = ~_mm256_setzero_si256();
          st.outputs[output] = create_circuit(&st, g_target[output], mask, bits, andnot, lut,
              randomize);
          if (st.outputs[output] == NO_GATE) {
            printf("No solution for output %d.\n", output);
            continue;
          }
          assert(ttable_equals(g_target[output], st.gates[st.outputs[output]].table));
          save_state(st);

          if (g_metric == GATES) {
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
    if (g_metric == GATES) {
      printf("Found %d state%s with %d gates.\n", num_out_states,
          num_out_states == 1 ? "" : "s", max_gates - 8);
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
  MPI_Bcast(&work, sizeof(work), MPI_BYTE, 0, MPI_COMM_WORLD);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  bool output_dot = false;
  bool output_c = false;
  bool lut_graph = false;
  bool andnot = false;
  bool randomize = true;
  char fname[1000];
  char gfname[1000];
  char sboxfname[1000];
  int oneoutput = -1;
  int permute = 0;
  int iterations = 1;
  int c;
  char *opts = "b:c:d:g:hi:lno:p:s";

  strcpy(fname, "");
  strcpy(gfname, "");
  strcpy(sboxfname, "");

  while ((c = getopt(argc, argv, opts)) != -1) {
    switch (c) {
      case 'b':
        if (strlen(optarg) >= 1000) {
          fprintf(stderr, "Error: File name too long.\n");
          MPI_Finalize();
          return 1;
        }
        strcpy(sboxfname, optarg);
        break;
      case 'c':
        if (rank != 0) {
          MPI_Finalize();
          return 0;
        }
        output_c = true;
        if (strlen(optarg) >= 1000) {
          fprintf(stderr, "Error: File name too long.\n");
          MPI_Finalize();
          return 1;
        }
        strcpy(fname, optarg);
        break;
      case 'd':
        if (rank != 0) {
          MPI_Finalize();
          return 0;
        }
        output_dot = true;
        if (strlen(optarg) >= 1000) {
          fprintf(stderr, "Error: File name too long.\n");
          MPI_Finalize();
          return 1;
        }
        strcpy(fname, optarg);
        break;
      case 'g':
        if (strlen(optarg) >= 1000) {
          fprintf(stderr, "Error: File name too long.\n");
          MPI_Finalize();
          return 1;
        }
        strcpy(gfname, optarg);
        break;
      case 'h':
        if (rank != 0) {
          MPI_Finalize();
          return 0;
        }
        printf(
            "-b file   Target S-box definition.\n"
            "-c file   Output C function.\n"
            "-d file   Output DOT digraph.\n"
            "-g file   Load graph from file as initial state. (For use with -o.)\n"
            "-h        Display this help.\n"
            "-i n      Do n iterations per step.\n"
            "-l        Generate LUT graph.\n"
            "-n        Use ANDNOT gates.\n"
            "-o n      Generate one-output graph for output n.\n"
            "-p value  Permute sbox by XORing input with value.\n"
            "-s        Use SAT metric.\n");
        MPI_Finalize();
        return 0;
      case 'i':
        iterations = atoi(optarg);
        if (iterations < 1) {
          fprintf(stderr, "Bad iterations value: %s\n", optarg);
        }
        break;
      case 'l':
        lut_graph = true;
        break;
      case 'n':
        andnot = true;
        break;
      case 'o':
        oneoutput = atoi(optarg);
        if (oneoutput < 0 || oneoutput > 7) {
          fprintf(stderr, "Bad output value: %s\n", optarg);
          MPI_Finalize();
          return 1;
        }
        break;
      case 'p':
        permute = atoi(optarg);
        if (permute < 0 || permute > 255) {
          fprintf(stderr, "Bad permutation value: %s\n", optarg);
          MPI_Finalize();
          return 1;
        }
        break;
      case 's':
        g_metric = SAT;
        break;
      default:
        MPI_Finalize();
        return 1;
    }
  }

  if (output_c && output_dot) {
    fprintf(stderr, "Cannot combine c and d options.\n");
    MPI_Finalize();
    return 1;
  }

  if (lut_graph && g_metric == SAT) {
    fprintf(stderr, "SAT metric can not be combined with LUT graph generation.\n");
    MPI_Finalize();
    return 1;
  }

  if (output_c || output_dot) {
    state st;
    if (!load_state(fname, &st)) {
      fprintf(stderr, "Error when reading state file.\n");
      MPI_Finalize();
      return 1;
    }
    if (output_c) {
      if (!print_c_function(st)) {
        MPI_Finalize();
        return 1;
      }
    } else {
      print_digraph(st);
    }
    MPI_Finalize();
    return 0;
  }

  if (rank != 0) {
    mpi_worker();
    MPI_Finalize();
    return 0;
  }

  if (strlen(sboxfname) == 0) {
    fprintf(stderr, "No target S-box file name argument.\n");
    stop_workers();
    return 1;
  }

  uint8_t target_sbox[256];
  int sbox_inp = 0;
  int ret;
  uint32_t input;
  FILE *sboxfp = fopen(sboxfname, "r");
  if (sboxfp == NULL) {
    fprintf(stderr, "Error when opening target S-box file.\n");
    stop_workers();
    return 1;
  }
  while ((ret = fscanf(sboxfp, " %x", &input)) > 0 && ret != EOF && sbox_inp < 256 && input < 256) {
    target_sbox[sbox_inp++] = input;
  }
  fclose(sboxfp);
  if (sbox_inp != 256) {
    fprintf(stderr, "Bad number of items in target S-box.\n");
    stop_workers();
    return 1;
  }

  if (permute == 0) {
    memcpy(g_sbox_enc, target_sbox, 256 * sizeof(uint8_t));
  } else {
    for (int i = 0; i < 256; i++) {
      g_sbox_enc[i] = target_sbox[i ^ (uint8_t)permute];
    }
  }

  /* Generate truth tables for all output bits of the target sbox. */
  for (uint8_t i = 0; i < 8; i++) {
    g_target[i] = generate_target(i, true);
  }

  state st;
  if (strlen(gfname) == 0) {
    st.max_sat_metric = INT_MAX;
    st.sat_metric = 0;
    st.max_gates = MAX_GATES;
    st.num_gates = 8;
    for (uint8_t i = 0; i < 8; i++) {
      st.gates[i].type = IN;
      st.gates[i].table = generate_target(i, false);
      st.gates[i].in1 = NO_GATE;
      st.gates[i].in2 = NO_GATE;
      st.gates[i].in3 = NO_GATE;
      st.gates[i].function = 0;
      st.outputs[i] = NO_GATE;
    }
  } else if (!load_state(gfname, &st)) {
    MPI_Finalize();
    return 1;
  } else {
    printf("Loaded %s.\n", gfname);
  }

  if (oneoutput != -1) {
    generate_graph_one_output(andnot, lut_graph, randomize, iterations, oneoutput, st);
  } else {
    generate_graph(andnot, lut_graph, randomize, iterations, st);
  }

  stop_workers();
  MPI_Finalize();

  return 0;
}
