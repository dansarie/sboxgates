/* sboxgates.c

   Program for finding low gate count implementations of S-boxes.
   The algorithm used is described in Kwan, Matthew: "Reducing the Gate Count of Bitslice DES."
   IACR Cryptology ePrint Archive 2000 (2000): 51. Improvements from
   SBOXDiscovery (https://github.com/DeepLearningJohnDoe/SBOXDiscovery) have been added.

   Copyright (c) 2016-2017 Marcus Dansarie

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
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <x86intrin.h>
#include "state.h"

#ifdef USE_MPI
#include <mpi.h>
#define MPI_FINALIZE() MPI_Finalize()
#else
#define MPI_FINALIZE()
#endif

#ifdef USE_MPI
typedef struct {
  state st;
  ttable target;
  ttable mask;
  bool quit;
} mpi_work;
#endif

const uint8_t g_lattice_sbox[] = {
    0x9c, 0xf2, 0x14, 0xc1, 0x8e, 0xcb, 0xb2, 0x65, 0x97, 0x7a, 0x60, 0x17, 0x92, 0xf9, 0x78, 0x41,
    0x07, 0x4c, 0x67, 0x6d, 0x66, 0x4a, 0x30, 0x7d, 0x53, 0x9d, 0xb5, 0xbc, 0xc3, 0xca, 0xf1, 0x04,
    0x03, 0xec, 0xd0, 0x38, 0xb0, 0xed, 0xad, 0xc4, 0xdd, 0x56, 0x42, 0xbd, 0xa0, 0xde, 0x1b, 0x81,
    0x55, 0x44, 0x5a, 0xe4, 0x50, 0xdc, 0x43, 0x63, 0x09, 0x5c, 0x74, 0xcf, 0x0e, 0xab, 0x1d, 0x3d,
    0x6b, 0x02, 0x5d, 0x28, 0xe7, 0xc6, 0xee, 0xb4, 0xd9, 0x7c, 0x19, 0x3e, 0x5e, 0x6c, 0xd6, 0x6e,
    0x2a, 0x13, 0xa5, 0x08, 0xb9, 0x2d, 0xbb, 0xa2, 0xd4, 0x96, 0x39, 0xe0, 0xba, 0xd7, 0x82, 0x33,
    0x0d, 0x5f, 0x26, 0x16, 0xfe, 0x22, 0xaf, 0x00, 0x11, 0xc8, 0x9e, 0x88, 0x8b, 0xa1, 0x7b, 0x87,
    0x27, 0xe6, 0xc7, 0x94, 0xd1, 0x5b, 0x9b, 0xf0, 0x9f, 0xdb, 0xe1, 0x8d, 0xd2, 0x1f, 0x6a, 0x90,
    0xf4, 0x18, 0x91, 0x59, 0x01, 0xb1, 0xfc, 0x34, 0x3c, 0x37, 0x47, 0x29, 0xe2, 0x64, 0x69, 0x24,
    0x0a, 0x2f, 0x73, 0x71, 0xa9, 0x84, 0x8c, 0xa8, 0xa3, 0x3b, 0xe3, 0xe9, 0x58, 0x80, 0xa7, 0xd3,
    0xb7, 0xc2, 0x1c, 0x95, 0x1e, 0x4d, 0x4f, 0x4e, 0xfb, 0x76, 0xfd, 0x99, 0xc5, 0xc9, 0xe8, 0x2e,
    0x8a, 0xdf, 0xf5, 0x49, 0xf3, 0x6f, 0x8f, 0xe5, 0xeb, 0xf6, 0x25, 0xd5, 0x31, 0xc0, 0x57, 0x72,
    0xaa, 0x46, 0x68, 0x0b, 0x93, 0x89, 0x83, 0x70, 0xef, 0xa4, 0x85, 0xf8, 0x0f, 0xb3, 0xac, 0x10,
    0x62, 0xcc, 0x61, 0x40, 0xf7, 0xfa, 0x52, 0x7f, 0xff, 0x32, 0x45, 0x20, 0x79, 0xce, 0xea, 0xbe,
    0xcd, 0x15, 0x21, 0x23, 0xd8, 0xb6, 0x0c, 0x3f, 0x54, 0x1a, 0xbf, 0x98, 0x48, 0x3a, 0x75, 0x77,
    0x2b, 0xae, 0x36, 0xda, 0x7e, 0x86, 0x35, 0x51, 0x05, 0x12, 0xb8, 0xa6, 0x9a, 0x2c, 0x06, 0x4b};

uint8_t g_sbox_enc[256];

ttable g_target[8];       /* Truth tables for the output bits of the sbox. */
metric g_metric = GATES;  /* Metric that should be used when selecting between two solutions. */

/* Prints a truth table to the console. Used for debugging. */
void print_ttable(ttable tbl) {
  uint64_t vec[4];
  _mm256_storeu_si256((ttable*)vec, tbl);
  uint64_t *var = &vec[0];
  for (uint16_t i = 0; i < 256; i++) {
    if (i == 64) {
      var = &vec[1];
    } else if (i == 128) {
      var = &vec[2];
    } else if (i == 192) {
      var = &vec[3];
    }
    if (i != 0 && i % 16 == 0) {
      printf("\n");
    }
    printf("%" PRIu64, (*var >> (i % 64)) & 1);
  }
  printf("\n");
}

/* Test two truth tables for equality. */
static inline bool ttable_equals(const ttable in1, const ttable in2) {
  ttable res = in1 ^ in2;
  return _mm256_testz_si256(res, res);
}

/* Performs a masked test for equality. Only bits set to 1 in the mask will be tested. */
static inline bool ttable_equals_mask(const ttable in1, const ttable in2, const ttable mask) {
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

static inline ttable generate_lut_ttable(const uint8_t function, const ttable in1, const ttable in2,
    const ttable in3) {
  ttable ret = _mm256_setzero_si256();
  if ((function & 1) != 0) {
    ret |= ~in1 & ~in2 & ~in3;
  }
  if ((function & 2) != 0) {
    ret |= ~in1 & ~in2 & in3;
  }
  if ((function & 4) != 0) {
    ret |= ~in1 & in2 & ~in3;
  }
  if ((function & 8) != 0) {
    ret |= ~in1 & in2 & in3;
  }
  if ((function & 16) != 0) {
    ret |= in1 & ~in2 & ~in3;
  }
  if ((function & 32) != 0) {
    ret |= in1 & ~in2 & in3;
  }
  if ((function & 64) != 0) {
    ret |= in1 & in2 & ~in3;
  }
  if ((function & 128) != 0) {
    ret |= in1 & in2 & in3;
  }
  return ret;
}

static inline void generate_lut_ttables(const ttable in1, const ttable in2, const ttable in3,
    ttable *out) {
  for (int func = 0; func < 256; func++) {
    out[func] = generate_lut_ttable(func, in1, in2, in3);
  }
}

static inline uint64_t xorshift1024() {
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

static inline bool get_lut_function(const ttable in1, const ttable in2, const ttable in3,
    const ttable target, const ttable mask, const bool randomize, uint8_t *func) {
  *func = 0;
  uint8_t tableset = 0;

  uint64_t in1_v[4];
  uint64_t in2_v[4];
  uint64_t in3_v[4];
  uint64_t target_v[4];
  uint64_t mask_v[4];

  _mm256_storeu_si256((ttable*)in1_v, in1);
  _mm256_storeu_si256((ttable*)in2_v, in2);
  _mm256_storeu_si256((ttable*)in3_v, in3);
  _mm256_storeu_si256((ttable*)target_v, target);
  _mm256_storeu_si256((ttable*)mask_v, mask);

  for (int v = 0; v < 4; v++) {
    for (int i = 0; i < 64; i++) {
      if (mask_v[v] & 1) {
        uint8_t temp = ((in1_v[v] & 1) << 2) | ((in2_v[v] & 1) << 1) | (in3_v[v] & 1);
        if ((tableset & (1 << temp)) == 0) {
          *func |= (target_v[v] & 1) << temp;
          tableset |= 1 << temp;
        } else if ((*func & (1 << temp)) != ((target_v[v] & 1) << temp)) {
          return false;
        }
      }
      target_v[v] >>= 1;
      mask_v[v] >>= 1;
      in1_v[v] >>= 1;
      in2_v[v] >>= 1;
      in3_v[v] >>= 1;
    }
  }

  /* Randomize don't-cares in table. */
  if (randomize && tableset != 0xff) {
    *func |= ~tableset & (uint8_t)xorshift1024();
  }

  return true;
}

static inline bool check_3lut_possible(const ttable target, const ttable mask, const ttable t1,
    const ttable t2, const ttable t3) {
  ttable match = _mm256_setzero_si256();
  ttable tt1 = ~t1;
  for (uint8_t i = 0; i < 2; i++) {
    ttable tt2 = ~t2;
    for (uint8_t k = 0; k < 2; k++) {
      ttable tt3 = ~t3;
      for (uint8_t m = 0; m < 2; m++) {
        ttable r = tt1 & tt2 & tt3;
        if (ttable_equals_mask(target & r, r, mask)) {
          match |= r;
        } else if (!_mm256_testz_si256(target & r & mask, target & r & mask)) {
          return false;
        }
        tt3 = ~tt3;
      }
      tt2 = ~tt2;
    }
    tt1 = ~tt1;
  }
  return ttable_equals_mask(target, match, mask);
}

static inline bool check_5lut_possible(const ttable target, const ttable mask, const ttable t1,
    const ttable t2, const ttable t3, const ttable t4, const ttable t5) {
  ttable match = _mm256_setzero_si256();
  ttable tt1 = ~t1;
  for (uint8_t i = 0; i < 2; i++) {
    ttable tt2 = ~t2;
    for (uint8_t k = 0; k < 2; k++) {
      ttable tt3 = ~t3;
      for (uint8_t m = 0; m < 2; m++) {
        ttable tt4 = ~t4;
        for (uint8_t o = 0; o < 2; o++) {
          ttable tt5 = ~t5;
          for (uint8_t q = 0; q < 2; q++) {
            ttable r = tt1 & tt2 & tt3 & tt4 & tt5;
            if (ttable_equals_mask(target & r, r, mask)) {
              match |= r;
            } else if (!_mm256_testz_si256(target & r & mask, target & r & mask)) {
              return false;
            }
            tt5 = ~tt5;
          }
          tt4 = ~tt4;
        }
        tt3 = ~tt3;
      }
      tt2 = ~tt2;
    }
    tt1 = ~tt1;
  }
  return ttable_equals_mask(target, match, mask);
}

static inline bool check_7lut_possible(const ttable target, const ttable mask, const ttable t1,
    const ttable t2, const ttable t3, const ttable t4, const ttable t5, const ttable t6,
    const ttable t7) {
  ttable match = _mm256_setzero_si256();
  ttable tt1 = ~t1;
  for (uint8_t i = 0; i < 2; i++) {
    ttable tt2 = ~t2;
    for (uint8_t k = 0; k < 2; k++) {
      ttable tt3 = ~t3;
      for (uint8_t m = 0; m < 2; m++) {
        ttable tt4 = ~t4;
        for (uint8_t o = 0; o < 2; o++) {
          ttable tt5 = ~t5;
          for (uint8_t q = 0; q < 2; q++) {
            ttable tt6 = ~t6;
            for (uint8_t s = 0; s < 2; s++) {
              ttable tt7 = ~t7;
              for (uint8_t u = 0; u < 2; u++) {
                ttable x = tt1 & tt2 & tt3 & tt4 & tt5 & tt6 & tt7;
                if (ttable_equals_mask(target & x, x, mask)) {
                  match |= x;
                } else if (!_mm256_testz_si256(target & x & mask, target & x & mask)) {
                  return false;
                }
                tt7 = ~tt7;
              }
              tt6 = ~tt6;
            }
            tt5 = ~tt5;
          }
          tt4 = ~tt4;
        }
        tt3 = ~tt3;
      }
      tt2 = ~tt2;
    }
    tt1 = ~tt1;
  }
  return ttable_equals_mask(target, match, mask);
}

#ifdef USE_MPI
static inline int64_t n_choose_k(int n, int k) {
  assert(n > 0);
  assert(k >= 0);
  int64_t ret = 1;
  for (int i = 1; i <= k; i++) {
    ret *= (n - i + 1);
    ret /= i;
  }
  return ret;
}

/* Generates the nth combination of num_gates choose t gates numbered first, first + 1, ...
   Return combination in ret[t]. */
static void get_nth_combination(int64_t n, int num_gates, int t, gatenum first,
    gatenum *ret) {
  assert(ret != NULL);
  assert(t < num_gates);

  if (t == 0) {
    return;
  }

  ret[0] = first;

  for (int i = 0; i < num_gates; i++) {
    if (n == 0) {
      for (int k = 1; k < t; k++) {
        ret[k] = ret[0] + k;
      }
      return;
    }
    int64_t nck = n_choose_k(num_gates - i - 1, t - 1);
    if (n < nck) {
      get_nth_combination(n, num_gates - ret[0] + first - 1, t - 1, ret[0] + 1, ret + 1);
      return;
    }
    ret[0] += 1;
    n -= nck;
  }
  assert(0);
}

/* Creates the next combination of t numbers from the set 0, 1, ..., max - 1. */
static inline void next_combination(gatenum *combination, int t, int max) {
  int i = t - 1;
  while (i >= 0) {
    if (combination[i] + t - i < max) {
      break;
    }
    i--;
  }
  if (i < 0) {
    return;
  }
  combination[i] += 1;
  for (int k = i + 1; k < t; k++) {
    combination[k] = combination[k - 1] + 1;
  }
}

static bool get_search_result(uint16_t *ret, int *quit_msg, MPI_Request *recv_req,
    MPI_Request *send_req) {

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Request reqs[2] = {*recv_req, MPI_REQUEST_NULL};
  MPI_Ibarrier(MPI_COMM_WORLD, &reqs[1]);
  if (rank == 0) {
    if (*recv_req == MPI_REQUEST_NULL) {
      for (int i = 1; i < size; i++) {
        MPI_Send(quit_msg, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
      }
    }
    int index;
    MPI_Waitany(2, reqs, &index, MPI_STATUSES_IGNORE);
    if (index == 0) { /* Received 'found' message. */
      for (int i = 1; i < size; i++) {
        MPI_Send(quit_msg, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
      }
      MPI_Wait(&reqs[1], MPI_STATUS_IGNORE);
    } else if (*quit_msg == -1) { /* No worker found a LUT. */
      for (int i = 1; i < size; i++) {
        MPI_Send(quit_msg, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
      }
      MPI_Cancel(&reqs[0]);
      MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
    }
  } else {
    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
  }

  *recv_req = reqs[0];

  if (*quit_msg == -1) {
    assert(*send_req == MPI_REQUEST_NULL);
    MPI_Barrier(MPI_COMM_WORLD);
    return false;
  }

  MPI_Bcast(ret, 10, MPI_SHORT, *quit_msg, MPI_COMM_WORLD);

  if (*send_req == MPI_REQUEST_NULL) {
    MPI_Isend(&rank, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, send_req);
  }

  if (rank == 0) {
    for (int i = 0; i < size; i++) {
      if (i == *quit_msg) {
        continue;
      }
      int buf;
      MPI_Recv(&buf, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
  MPI_Wait(send_req, MPI_STATUS_IGNORE);
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

static bool search_5lut(const state st, const ttable target, const ttable mask,
    uint16_t *ret) {
  assert(ret != NULL);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  uint8_t func_order[256];
  for (int i = 0; i < 256; i++) {
    func_order[i] = i;
  }
  /* Fisher-Yates shuffle. */
  for (int i = 0; i < 256; i++) {
    uint64_t j = xorshift1024() % (i + 1);
    uint8_t t = func_order[i];
    func_order[i] = func_order[j];
    func_order[j] = t;
  }

  /* Determine this rank's work. */
  uint64_t search_space_size = n_choose_k(st.num_gates, 5);
  uint64_t worker_space_size = search_space_size / size;
  uint64_t remainder = search_space_size - worker_space_size * size;
  uint64_t start_n;
  uint64_t stop_n;
  if (rank < remainder) {
    start_n = (worker_space_size + 1) * rank;
    stop_n = start_n + worker_space_size + 1;
  } else {
    start_n = (worker_space_size + 1) * remainder + worker_space_size * (rank - remainder);
    stop_n = start_n + worker_space_size;
  }
  gatenum nums[5] = {NO_GATE, NO_GATE, NO_GATE, NO_GATE, NO_GATE};
  get_nth_combination(start_n, st.num_gates, 5, 0, nums);

  ttable tt[5] = {st.gates[nums[0]].table, st.gates[nums[1]].table, st.gates[nums[2]].table,
      st.gates[nums[3]].table, st.gates[nums[4]].table};
  gatenum cache_set[3] = {NO_GATE, NO_GATE, NO_GATE};
  ttable cache[256];

  memset(ret, 0, sizeof(uint16_t) * 10);

  MPI_Request recv_req = MPI_REQUEST_NULL;
  MPI_Request send_req = MPI_REQUEST_NULL;
  int quit_msg = -1;

  if (rank == 0) {
    MPI_Irecv(&quit_msg, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &recv_req);
  } else {
    MPI_Irecv(&quit_msg, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &recv_req);
  }

  bool quit = false;
  for (uint64_t i = start_n; !quit && i < stop_n; i++) {
    if (check_5lut_possible(target, mask, tt[0], tt[1], tt[2], tt[3], tt[4])) {
      if (cache_set[0] != nums[0] || cache_set[1] != nums[1] || cache_set[2] != nums[2]) {
        generate_lut_ttables(tt[0], tt[1], tt[2], cache);
        cache_set[0] = nums[0];
        cache_set[1] = nums[1];
        cache_set[2] = nums[2];
      }

      for (uint16_t fo = 0; !quit && fo < 256; fo++) {
        uint8_t func_outer = func_order[fo];
        ttable t_outer = cache[func_outer];
        uint8_t func_inner;
        if (!get_lut_function(t_outer, tt[3], tt[4], target, mask, true, &func_inner)) {
          continue;
        }
        ttable t_inner = generate_lut_ttable(func_inner, t_outer, tt[3], tt[4]);
        assert(ttable_equals_mask(target, t_inner, mask));
        ret[0] = func_outer;
        ret[1] = func_inner;
        ret[2] = nums[0];
        ret[3] = nums[1];
        ret[4] = nums[2];
        ret[5] = nums[3];
        ret[6] = nums[4];
        ret[7] = 0;
        ret[8] = 0;
        ret[9] = 0;
        assert(send_req == MPI_REQUEST_NULL);
        MPI_Isend(&rank, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &send_req);
        quit = true;
      }
    }
    if (!quit) {
      int flag;
      MPI_Test(&recv_req, &flag, MPI_STATUS_IGNORE);
      if (flag) {
        break;
      }
      next_combination(nums, 5, st.num_gates);
    }
  }

  return get_search_result(ret, &quit_msg, &recv_req, &send_req);
}

static bool search_7lut(const state st, const ttable target, const ttable mask,
    uint16_t *ret) {
  assert(ret != NULL);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* Determine this rank's work. */
  uint64_t search_space_size = n_choose_k(st.num_gates, 7);
  uint64_t worker_space_size = search_space_size / size;
  uint64_t remainder = search_space_size - worker_space_size * size;
  uint64_t start;
  uint64_t stop;
  if (rank < remainder) {
    start = (worker_space_size + 1) * rank;
    stop = start + worker_space_size + 1;
  } else {
    start = (worker_space_size + 1) * remainder + worker_space_size * (rank - remainder);
    stop = start + worker_space_size;
  }
  gatenum nums[7];
  get_nth_combination(start, st.num_gates, 7, 0, nums);

  ttable tt[7] = {st.gates[nums[0]].table, st.gates[nums[1]].table, st.gates[nums[2]].table,
      st.gates[nums[3]].table, st.gates[nums[4]].table, st.gates[nums[5]].table,
      st.gates[nums[6]].table};

  /* Filter out the gate combinations where a 7LUT is possible. */
  gatenum *result = malloc(7 * 100000 * sizeof(gatenum));
  assert(result != NULL);
  int p = 0;
  for (uint64_t i = start; i < stop; i++) {
    if (check_7lut_possible(target, mask, tt[0], tt[1], tt[2], tt[3], tt[4], tt[5], tt[6])) {
      result[p++] = nums[0];
      result[p++] = nums[1];
      result[p++] = nums[2];
      result[p++] = nums[3];
      result[p++] = nums[4];
      result[p++] = nums[5];
      result[p++] = nums[6];
    }
    if (p >= 7 * 100000) {
      break;
    }
    next_combination(nums, 7, st.num_gates);
  }

  /* Gather the number of hits for each rank.*/
  int rank_nums[size];
  MPI_Allgather(&p, 1, MPI_INT, rank_nums, 1, MPI_INT, MPI_COMM_WORLD);
  assert(rank_nums[0] % 7 == 0);
  int tsize = rank_nums[0];
  int offsets[size];
  offsets[0] = 0;
  for (int i = 1; i < size; i++) {
    assert(rank_nums[i] % 7 == 0);
    tsize += rank_nums[i];
    offsets[i] = offsets[i - 1] + rank_nums[i - 1];
  }

  gatenum *lut_list = malloc(tsize * sizeof(gatenum));
  assert(lut_list != NULL);

  /* Get all hits. */
  MPI_Allgatherv(result, p, MPI_UINT16_T, lut_list, rank_nums, offsets, MPI_UINT16_T,
      MPI_COMM_WORLD);
  free(result);
  result = NULL;

  /* Calculate rank's work chunk. */
  worker_space_size = (tsize / 7) / size;
  remainder = (tsize / 7) - worker_space_size * size;
  if (rank < remainder) {
    start = (worker_space_size + 1) * rank;
    stop  = start + worker_space_size + 1;
  } else {
    start = (worker_space_size + 1) * remainder + worker_space_size * (rank - remainder);
    stop = start + worker_space_size;
  }

  uint8_t outer_func_order[256];
  uint8_t middle_func_order[256];
  for (int i = 0; i < 256; i++) {
    outer_func_order[i] = middle_func_order[i] = i;
  }

  /* Fisher-Yates shuffle the function search orders. */
  for (int i = 0; i < 256; i++) {
    uint64_t oj = xorshift1024() % (i + 1);
    uint64_t mj = xorshift1024() % (i + 1);
    uint8_t ot = outer_func_order[i];
    uint8_t mt = middle_func_order[i];
    outer_func_order[i] = outer_func_order[oj];
    middle_func_order[i] = middle_func_order[mj];
    outer_func_order[oj] = ot;
    middle_func_order[mj] = mt;
  }
  int outer_cache_set = 0;
  int middle_cache_set = 0;
  ttable outer_cache[256];
  ttable middle_cache[256];
  memset(ret, 0, 10 * sizeof(uint16_t));

  MPI_Request recv_req = MPI_REQUEST_NULL;
  MPI_Request send_req = MPI_REQUEST_NULL;
  int quit_msg = -1;

  if (rank == 0) {
    MPI_Irecv(&quit_msg, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &recv_req);
  } else {
    MPI_Irecv(&quit_msg, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &recv_req);
  }

  bool quit = false;
  for (int i = start; !quit && i < stop; i++) {
    const gatenum a = lut_list[7 * i];
    const gatenum b = lut_list[7 * i + 1];
    const gatenum c = lut_list[7 * i + 2];
    const gatenum d = lut_list[7 * i + 3];
    const gatenum e = lut_list[7 * i + 4];
    const gatenum f = lut_list[7 * i + 5];
    const gatenum g = lut_list[7 * i + 6];
    const ttable ta = st.gates[a].table;
    const ttable tb = st.gates[b].table;
    const ttable tc = st.gates[c].table;
    const ttable td = st.gates[d].table;
    const ttable te = st.gates[e].table;
    const ttable tf = st.gates[f].table;
    const ttable tg = st.gates[g].table;
    if (((uint64_t)a << 32 | (uint64_t)b << 16 | c) != outer_cache_set) {
      generate_lut_ttables(ta, tb, tc, outer_cache);
      outer_cache_set = (uint64_t)a << 32 | (uint64_t)b << 16 | c;
    }
    if (((uint64_t)d << 32 | (uint64_t)e << 16 | f) != middle_cache_set) {
      generate_lut_ttables(td, te, tf, middle_cache);
      middle_cache_set = (uint64_t)d << 32 | (uint64_t)e << 16 | f;
    }

    for (uint16_t fo = 0; !quit && fo < 256; fo++) {
      uint8_t func_outer = outer_func_order[fo];
      ttable t_outer = outer_cache[func_outer];
      for (uint16_t fm = 0; !quit && fm < 256; fm++) {
        uint8_t func_middle = middle_func_order[fm];
        ttable t_middle = middle_cache[func_middle];
        uint8_t func_inner;
        if (!get_lut_function(t_outer, t_middle, tg, target, mask, true, &func_inner)) {
          continue;
        }
        ttable t_inner = generate_lut_ttable(func_inner, t_outer, t_middle, tg);
        assert(ttable_equals_mask(target, t_inner, mask));
        ret[0] = func_outer;
        ret[1] = func_middle;
        ret[2] = func_inner;
        ret[3] = a;
        ret[4] = b;
        ret[5] = c;
        ret[6] = d;
        ret[7] = e;
        ret[8] = f;
        ret[9] = g;

        assert(send_req == MPI_REQUEST_NULL);
        MPI_Isend(&rank, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &send_req);
        quit = true;
      }
    }
    if (!quit) {
      int flag;
      MPI_Test(&recv_req, &flag, MPI_STATUS_IGNORE);
      if (flag) {
        quit = true;
      }
    }
  }
  free(lut_list);
  return get_search_result(ret, &quit_msg, &recv_req, &send_req);
}
#endif

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

    #ifdef USE_MPI
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

    #else
    /* Look through all combinations of five gates in the circuit. For each combination, check if
       a combination of two of the possible 256 three bit Boolean functions as in
       LUT(LUT(a,b,c),d,e) produces the desired map. If so, add those LUTs and return the ID of the
       output LUT. */

    for (int i = 0; i < st->num_gates; i++) {
      const gatenum gi = gate_order[i];
      const ttable ta = st->gates[gi].table;
      for (int k = i + 1; k < st->num_gates; k++) {
        const gatenum gk = gate_order[k];
        const ttable tb = st->gates[gk].table;
        for (int m = k + 1; m < st->num_gates; m++) {
          const gatenum gm = gate_order[m];
          const ttable tc = st->gates[gm].table;
          bool cache_set = false;
          ttable cache[256];
          for (int o = m + 1; o < st->num_gates; o++) {
            const gatenum go = gate_order[o];
            const ttable td = st->gates[go].table;
            for (int q = o + 1; q < st->num_gates; q++) {
              const gatenum gq = gate_order[q];
              const ttable te = st->gates[gq].table;
              if (!check_5lut_possible(target, mask, ta, tb, tc, td, te)) {
                continue;
              }
              if (!cache_set) {
                generate_lut_ttables(ta, tb, tc, cache);
                cache_set = true;
              }
              for (int func_outer = 0; func_outer < 256; func_outer++) {
                ttable t_outer = cache[func_outer];
                uint8_t func_inner;
                if (!get_lut_function(t_outer, td, te, target, mask, randomize, &func_inner)) {
                  continue;
                }
                ttable t_inner = generate_lut_ttable(func_inner, t_outer, td, te);
                assert(ttable_equals_mask(target, t_inner, mask));
                return add_lut(st, func_inner, t_inner,
                    add_lut(st, func_outer, t_outer, gi, gk, gm), go, gq);
              }
            }
          }
        }
      }
    }

    /* Look through all combinations of seven gates in the circuit. For each combination, check if
       a combination of three of the possible 256 three bit Boolean functions as in
       LUT(LUT(a,b,c),LUT(d,e,f),g) produces the desired map. If so, add those LUTs and return the
       ID of the output LUT. */

    for (int i = 0; i < st->num_gates; i++) {
      const gatenum gi = gate_order[i];
      const ttable ta = st->gates[gi].table;
      for (int k = i + 1; k < st->num_gates; k++) {
        const gatenum gk = gate_order[k];
        const ttable tb = st->gates[gk].table;
        for (int m = k + 1; m < st->num_gates; m++) {
          const gatenum gm = gate_order[m];
          const ttable tc = st->gates[gm].table;
          bool outer_cache_set = false;
          ttable outer_cache[256];
          for (int o = m + 1; o < st->num_gates; o++) {
            const gatenum go = gate_order[o];
            const ttable td = st->gates[go].table;
            for (int q = o + 1; q < st->num_gates; q++) {
              const gatenum gq = gate_order[q];
              const ttable te = st->gates[gq].table;
              for (int s = q + 1; s < st->num_gates; s++) {
                const gatenum gs = gate_order[s];
                const ttable tf = st->gates[gs].table;
                bool middle_cache_set = false;
                ttable middle_cache[256];
                for (int u = s + 1; u < st->num_gates; u++) {
                  const gatenum gu = gate_order[u];
                  const ttable tg = st->gates[gu].table;
                  if (!check_7lut_possible(target, mask, ta, tb, tc, td, te, tf, tg)) {
                    continue;
                  }
                  if (!outer_cache_set) {
                    generate_lut_ttables(ta, tb, tc, outer_cache);
                    outer_cache_set = true;
                  }
                  if (!middle_cache_set) {
                    generate_lut_ttables(td, te, tf, middle_cache);
                    middle_cache_set = true;
                  }
                  for (int func_outer = 0; func_outer < 256; func_outer++) {
                    ttable t_outer = outer_cache[func_outer];
                    for (int func_middle = 0; func_middle < 256; func_middle++) {
                      ttable t_middle = middle_cache[func_middle];
                      uint8_t func_inner;
                      if (!get_lut_function(t_outer, t_middle, tg, target, mask, randomize,
                          &func_inner)) {
                        continue;
                      }
                      ttable t_inner = generate_lut_ttable(func_inner, t_outer, t_middle, tg);
                      assert(ttable_equals_mask(target, t_inner, mask));
                      return add_lut(st, func_inner, t_inner,
                          add_lut(st, func_outer, t_outer, gi, gk, gm),
                          add_lut(st, func_middle, t_middle, go, gq, gs), gu);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    #endif

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
        assert(mux_out_and == NO_GATE || ttable_equals_mask(target, nst_and.gates[mux_out_and].table, mask));
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
        assert(mux_out_or == NO_GATE || ttable_equals_mask(target, nst_or.gates[mux_out_or].table, mask));
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

#ifdef USE_MPI

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
#endif

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

/* Prints a gate network to stdout in Graphviz dot format. */
void print_digraph(const state st) {
  printf("digraph sbox {\n");
  assert(st.num_gates < MAX_GATES);
  for (int gt = 0; gt < st.num_gates; gt++) {
    char gatename[10];
    switch (st.gates[gt].type) {
      case IN:
        sprintf(gatename, "IN %d", gt);
        break;
      case NOT:
        strcpy(gatename, "NOT");
        break;
      case AND:
        strcpy(gatename, "AND");
        break;
      case OR:
        strcpy(gatename, "OR");
        break;
      case XOR:
        strcpy(gatename, "XOR");
        break;
      case ANDNOT:
        strcpy(gatename, "ANDNOT");
        break;
      case LUT:
        sprintf(gatename, "0x%02x", st.gates[gt].function);
        break;
      default:
        assert(0);
    }
    printf("  gt%d [label=\"%s\"];\n", gt, gatename);
  }
  for (int gt = 8; gt < st.num_gates; gt++) {
    if (st.gates[gt].in1 != NO_GATE) {
      printf("  gt%" PRIgatenum " -> gt%d;\n", st.gates[gt].in1, gt);
    }
    if (st.gates[gt].in2 != NO_GATE) {
      printf("  gt%" PRIgatenum " -> gt%d;\n", st.gates[gt].in2, gt);
    }
    if (st.gates[gt].in3 != NO_GATE) {
      printf("  gt%" PRIgatenum " -> gt%d;\n", st.gates[gt].in3, gt);
    }
  }
  for (uint8_t i = 0; i < 8; i++) {
    if (st.outputs[i] != NO_GATE) {
      printf("  gt%" PRIgatenum " -> out%" PRIu8 ";\n", st.outputs[i], i);
    }
  }
  printf("}\n");
}

/* Called by print_c_function to get variable names. */
static bool get_c_variable_name(const state st, const gatenum gate, char *buf) {
  if (gate < 8) {
    sprintf(buf, "in.b%" PRIgatenum, gate);
    return false;
  }
  for (uint8_t i = 0; i < 8; i++) {
    if (st.outputs[i] == gate) {
      sprintf(buf, "out%d", i);
      return false;
    }
  }
  sprintf(buf, "var%" PRIgatenum, gate);
  return true;
}

/* Converts the given state gate network to a C function and prints it to stdout. */
static void print_c_function(const state st) {
  const char TYPE[] = "int ";
  char buf[10];
  printf("__device__ __forceinline__ int s0(eightbits in) {\n");
  for (int gate = 8; gate < st.num_gates; gate++) {
    bool ret = get_c_variable_name(st, gate, buf);
    printf("  %s%s = ", ret == true ? TYPE : "", buf);
    get_c_variable_name(st, st.gates[gate].in1, buf);
    if (st.gates[gate].type == NOT) {
      printf("~%s;\n", buf);
      continue;
    }
    if (st.gates[gate].type == ANDNOT) {
      printf("~");
    }
    printf("%s ", buf);
    switch (st.gates[gate].type) {
      case AND:
      case ANDNOT:
        printf("& ");
        break;
      case OR:
        printf("| ");
        break;
      case XOR:
        printf("^ ");
        break;
      default:
        assert(false);
    }
    get_c_variable_name(st, st.gates[gate].in2, buf);
    printf("%s;\n", buf);
  }
  printf("}\n");
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

/* Called by main to generate a graph. */
void generate_graph(const bool andnot, const bool lut, const bool randomize, const int iterations,
    const state st) {
  int num_start_states = 1;
  state start_states[20];
  start_states[0] = st;

  /* Build the gate network one output at a time. After every added output, select the gate network
     or network with the least amount of gates and add another. */
  while (1) {
    gatenum max_gates = MAX_GATES;
    int max_sat_metric = INT_MAX;
    state out_states[20];
    int num_out_states = 0;

    /* Count the outputs already present in the first of the start states. All start states will
       have the same number of outputs. */
    uint8_t num_outputs = 0;
    for (uint8_t i = 0; i < 8; i++) {
      if (start_states[0].outputs[i] != NO_GATE) {
        num_outputs += 1;
      }
    }
    if (num_outputs >= 8) {
      /* If the input gate network has eight outputs, there is nothing more to do. */
      printf("Done.\n");
      break;
    }
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
              assert(num_out_states < 20); /* Very unlikely, but not impossible. */
              out_states[num_out_states++] = st;
            }
          } else {
            if (max_sat_metric > st.sat_metric) {
              max_sat_metric = st.sat_metric;
              num_out_states = 0;
            }
            if (st.sat_metric <= max_sat_metric) {
              assert(num_out_states < 20); /* Very unlikely, but not impossible. */
              out_states[num_out_states++] = st;
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

int main(int argc, char **argv) {

  #ifdef USE_MPI
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  #endif

  bool output_dot = false;
  bool output_c = false;
  bool lut_graph = false;
  bool andnot = false;
  #ifdef USE_MPI
  bool randomize = true;
  #else
  bool randomize = false;
  #endif
  char fname[1000];
  char gfname[1000];
  int oneoutput = -1;
  int permute = 0;
  int iterations = 1;
  int c;
  #ifdef USE_MPI
  char *opts = "c:d:g:hi:lno:p:s";
  #else
  char *opts = "c:d:g:hi:lno:p:rs";
  #endif

  strcpy(fname, "");
  strcpy(gfname, "");

  while ((c = getopt(argc, argv, opts)) != -1) {
    switch (c) {
      case 'c':
        #ifdef USE_MPI
        if (rank != 0) {
          MPI_Finalize();
          return 0;
        }
        #endif
        output_c = true;
        if (strlen(optarg) >= 1000) {
          fprintf(stderr, "Error: File name too long.\n");
          MPI_FINALIZE();
          return 1;
        }
        strcpy(fname, optarg);
        break;
      case 'd':
        #ifdef USE_MPI
        if (rank != 0) {
          MPI_Finalize();
          return 0;
        }
        #endif
        output_dot = true;
        if (strlen(optarg) >= 1000) {
          fprintf(stderr, "Error: File name too long.\n");
          MPI_FINALIZE();
          return 1;
        }
        strcpy(fname, optarg);
        break;
      case 'g':
        if (strlen(optarg) >= 1000) {
          fprintf(stderr, "Error: File name too long.\n");
          MPI_FINALIZE();
          return 1;
        }
        strcpy(gfname, optarg);
        break;
      case 'h':
        #ifdef USE_MPI
        if (rank != 0) {
          MPI_Finalize();
          return 0;
        }
        #endif
        printf(
            "-c file   Output C function.\n"
            "-d file   Output DOT digraph.\n"
            "-g file   Load graph from file as initial state. (For use with -o.)\n"
            "-h        Display this help.\n"
            "-i n      Do n iterations per step.\n"
            "-l        Generate LUT graph.\n"
            "-n        Use ANDNOT gates.\n"
            "-o n      Generate one-output graph for output n.\n"
            "-p value  Permute sbox by XORing input with value.\n"
            #ifndef USE_MPI
            "-r        Enable randomization.\n"
            #endif
            "-s        Use SAT metric.\n");
        MPI_FINALIZE();
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
          MPI_FINALIZE();
          return 1;
        }
        break;
      case 'p':
        permute = atoi(optarg);
        if (permute < 0 || permute > 255) {
          fprintf(stderr, "Bad permutation value: %s\n", optarg);
          MPI_FINALIZE();
          return 1;
        }
        break;
      #ifndef USE_MPI
      case 'r':
        randomize = true;
        break;
      #endif
      case 's':
        g_metric = SAT;
        break;
      default:
        MPI_FINALIZE();
        return 1;
    }
  }

  if (output_c && output_dot) {
    fprintf(stderr, "Cannot combine c and d options.\n");
    MPI_FINALIZE();
    return 1;
  }

  if (lut_graph && g_metric == SAT) {
    fprintf(stderr, "SAT metric can not be combined with LUT graph generation.\n");
    MPI_FINALIZE();
    return 1;
  }

  if (output_c || output_dot) {
    state st;
    if (!load_state(fname, &st)) {
      fprintf(stderr, "Error when reading state file.\n");
      MPI_FINALIZE();
      return 1;
    }
    if (output_c) {
      for (int i = 0; i < st.num_gates; i++) {
        if (st.gates[i].type == LUT) {
          fprintf(stderr, "C output of graphs containing LUTs is not supported.\n");
          MPI_FINALIZE();
          return 1;
        }
      }
      print_c_function(st);
    } else {
      print_digraph(st);
    }
    MPI_FINALIZE();
    return 0;
  }

  if (permute == 0) {
    memcpy(g_sbox_enc, g_lattice_sbox, 256 * sizeof(uint8_t));
  } else {
    for (int i = 0; i < 256; i++) {
      g_sbox_enc[i] = g_lattice_sbox[i ^ (uint8_t)permute];
    }
  }

  /* Generate truth tables for all output bits of the target sbox. */
  for (uint8_t i = 0; i < 8; i++) {
    g_target[i] = generate_target(i, true);
  }

  #ifdef USE_MPI
  if (lut_graph) {
    if (rank != 0) {
      mpi_worker();
      MPI_Finalize();
      return 0;
    }
  }
  #endif

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
    MPI_FINALIZE();
    return 1;
  } else {
    printf("Loaded %s.\n", gfname);
  }

  if (oneoutput != -1) {
    generate_graph_one_output(andnot, lut_graph, randomize, iterations, oneoutput, st);
  } else {
    generate_graph(andnot, lut_graph, randomize, iterations, st);
  }

  #ifdef USE_MPI
  mpi_work work;
  work.quit = true;
  MPI_Bcast(&work, sizeof(work), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  #endif

  MPI_FINALIZE();
  return 0;
}
