/*
 * Program for finding low gate count implementations of S-boxes.
 * The algorithm used is described in Kwan, Matthew: "Reducing the Gate Count of Bitslice DES."
 * IACR Cryptology ePrint Archive 2000 (2000): 51.
 *
 * Copyright (c) 2016-2017 Marcus Dansarie
 */

#include <assert.h>
#include <inttypes.h>
#include <msgpack.h>
#include <msgpack/fbuffer.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <x86intrin.h>

#define MSGPACK_FORMAT_VERSION 2
#define MAX_GATES 500
#define NO_GATE ((gatenum)-1)
#define PRIgatenum PRIu16

typedef enum {IN, NOT, AND, OR, XOR, ANDNOT, LUT} gate_type;
typedef enum {GATES, SAT} metric;

typedef __m256i ttable; /* 256 bit truth table. */
typedef uint16_t gatenum;

typedef struct {
  ttable table;
  gate_type type;
  gatenum in1; /* Input 1 to the gate. NO_GATE for the inputs. */
  gatenum in2; /* Input 2 to the gate. NO_GATE for NOT gates and the inputs. */
  gatenum in3; /* Input 3 if LUT or NO_GATE. */
  uint8_t function; /* For LUTs. */
} gate;

typedef struct {
  int max_sat_metric;
  int sat_metric;
  gatenum max_gates;
  gatenum num_gates;  /* Current number of gates. */
  gatenum outputs[8]; /* Gate number of the respective output gates, or NO_GATE. */
  gate gates[MAX_GATES];
} state;

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
uint8_t g_verbosity = 0;  /* Verbosity level. Higher = more debugging messages. */
metric g_metric = GATES;  /* Metric that should be used when selecting between two solutions. */
FILE *g_rand_fp = NULL;   /* Pointer to /dev/urandom. */

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
  switch (type) {
    case NOT:
      break;
    case AND:
    case OR:
    case ANDNOT:
      st->sat_metric += 1;
      break;
    case XOR:
      st->sat_metric += 4;
      break;
    default:
      assert(0);
  }
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

static inline bool get_lut_function(const ttable in1, const ttable in2, const ttable in3,
    const ttable target, const ttable mask, uint8_t *func) {
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

/* Recursively builds the gate network. The numbered comments are references to Matthew Kwan's
   paper. */
static gatenum create_circuit(state *st, const ttable target, const ttable mask,
    const int8_t *inbits, const bool andnot, const bool lut, const bool randomize) {

  gatenum gate_order[MAX_GATES];
  for (int i = 0; i < st->num_gates; i++) {
    gate_order[i] = st->num_gates - 1 - i;
  }

  if (randomize) {
    assert(g_rand_fp != NULL);
    /* Fisher-Yates shuffle. With a 1024 bit PRNG state, we can theoretically get every
       permutation of lists with less than or equal to 170 elements. */
    uint64_t rand[16];
    if (fread(&rand, 16 * sizeof(uint64_t), 1, g_rand_fp) != 1) {
      fprintf(stderr, "Error when reading from /dev/urandom.\n");
    }
    int p = 0;
    for (uint32_t i = st->num_gates - 1; i > 0; i--) {
      /* xorshift1024* */
      uint64_t r0 = rand[p];
      p = (p + 1) & 15;
      uint64_t r1 = rand[p];
      r1 ^= r1 << 31;
      rand[p] = r1 ^ r0 ^ (r1 >> 11) ^ (r0 >> 30);
      uint32_t j = (rand[p] * 1181783497276652981U) % (i + 1);
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
      if (ttable_equals(mtarget, ti ^ tk)) {
        return add_xor_gate(st, gi, gk);
      }
      if (andnot) {
        if (ttable_equals_mask(target, ~ti & tk, mask)) {
          return add_andnot_gate(st, gi, gk);
        }
        if (ttable_equals_mask(target, ~tk & ti, mask)) {
          return add_andnot_gate(st, gk, gi);
        }
      }
    }
  }

  if (lut) {
    /* Look through all combinations of three gates in the circuit. For each combination, check if
       any of the 256 possible three bit boolean functions produces the desired map. If so, add that
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
          if (!get_lut_function(ta, tb, tc, target, mask, &func)) {
            continue;
          }
          ttable nt = generate_lut_ttable(func, ta, tb, tc);
          assert(ttable_equals_mask(target, nt, mask));
          return add_lut(st, func, nt, gi, gk, gm);
        }
      }
    }

    /* Look through all combinations of five gates in the circuit. For each combination, check if
       a combination of two of the possible 256 three bit boolean functions as in
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
                if (!get_lut_function(t_outer, td, te, target, mask, &func_inner)) {
                  continue;
                }
                ttable t_inner = generate_lut_ttable(func_inner, t_outer, td, te);
                assert(ttable_equals_mask(target, t_inner, mask));
                if (g_verbosity > 2) {
                  printf("Found 5LUT!\n");
                }
                return add_lut(st, func_inner, t_inner,
                    add_lut(st, func_outer, t_outer, gi, gk, gm), go, gq);
              }
            }
          }
        }
      }
    }

    /* Look through all combinations of seven gates in the circuit. For each combination, check if
       a combination of three of the possible 256 three bit boolean functions as in
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
                      if (!get_lut_function(t_outer, t_middle, tg, target, mask, &func_inner)) {
                        continue;
                      }
                      ttable t_inner = generate_lut_ttable(func_inner, t_outer, t_middle, tg);
                      assert(ttable_equals_mask(target, t_inner, mask));
                      if (g_verbosity > 2) {
                        printf("Found 7LUT!\n");
                      }
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
        if (ttable_equals_mask(target, ~(ti ^ tk), mask)) {
          return add_xnor_gate(st, gi, gk);
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
          if (ttable_equals(mtarget, iandk ^ tm)) {
            return add_and_xor_gate(st, gi, gk, gm);
          }
          if (ttable_equals(mtarget, iork | tm)) {
            return add_or_3_gate(st, gi, gk, gm);
          }
          if (ttable_equals(mtarget, iork & tm)) {
            return add_or_and_gate(st, gi, gk, gm);
          }
          if (ttable_equals(mtarget, iork ^ tm)) {
            return add_or_xor_gate(st, gi, gk, gm);
          }
          if (ttable_equals(mtarget, ixork ^ tm)) {
            return add_xor_3_gate(st, gi, gk, gm);
          }
          if (ttable_equals(mtarget, ixork | tm)) {
            return add_xor_or_gate(st, gi, gk, gm);
          }
          if (ttable_equals(mtarget, ixork & tm)) {
            return add_xor_and_gate(st, gi, gk, gm);
          }
          ttable iandm = ti & tm;
          if (ttable_equals(mtarget, iandm | tk)) {
            return add_and_or_gate(st, gi, gm, gk);
          }
          if (ttable_equals(mtarget, iandm ^ tk)) {
            return add_and_xor_gate(st, gi, gm, gk);
          }
          ttable kandm = tk & tm;
          if (ttable_equals(mtarget, kandm | ti)) {
            return add_and_or_gate(st, gk, gm, gi);
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
          ttable iorm = ti | tm;
          if (ttable_equals(mtarget, iorm & tk)) {
            return add_or_and_gate(st, gi, gm, gk);
          }
          if (ttable_equals(mtarget, iorm ^ tk)) {
            return add_or_xor_gate(st, gi, gm, gk);
          }
          ttable korm = tk | tm;
          if (ttable_equals(mtarget, korm & ti)) {
            return add_or_and_gate(st, gk, gm, gi);
          }
          if (ttable_equals(mtarget, korm ^ ti)) {
            return add_or_xor_gate(st, gk, gm, gi);
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
    if (lut) {
      nst = *st;
      gatenum fb = create_circuit(&nst, target, mask & ~fsel, next_inbits, andnot, true, randomize);
      if (fb == NO_GATE) {
        continue;
      }
      gatenum fc = create_circuit(&nst, target, mask & fsel, next_inbits, andnot, true, randomize);
      if (fc == NO_GATE) {
        continue;
      }

      if (fb == bit || fc == bit) {
        add_or_gate(&nst, fb, fc);
      } else {
        ttable mux_table = generate_lut_ttable(0xac, nst.gates[bit].table, nst.gates[fb].table,
            nst.gates[fc].table);
        add_lut(&nst, 0xac, mux_table, bit, fb, fc);
      }
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
      }

      state nst_or = *st;
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
        } else {
          nst = nst_or;
        }
      } else {
        if (mux_out_or == NO_GATE
            || (mux_out_and != NO_GATE && nst_and.sat_metric < nst_or.sat_metric)) {
          nst = nst_and;
        } else {
          nst = nst_or;
        }
      }
    }

    if (g_metric == GATES) {
      if (best.num_gates == 0 || nst.num_gates < best.num_gates) {
        best = nst;
      }
    } else {
      if (best.sat_metric == 0 || nst.sat_metric < best.sat_metric) {
        best = nst;
      }
    }
  }

  if (best.num_gates == 0) {
    return NO_GATE;
  }

  assert(ttable_equals_mask(target, best.gates[best.num_gates - 1].table, mask));
  *st = best;
  return best.num_gates - 1;
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

static inline uint32_t speck_round(uint16_t pt1, uint16_t pt2, uint16_t k1) {
  pt1 = (pt1 >> 7) | (pt1 << 9);
  pt1 += pt2;
  pt2 = (pt2 >> 14) | (pt2 << 2);
  pt1 ^= k1;
  pt2 ^= pt1;
  return (((uint32_t)pt1) << 16) | pt2;
}

/* Generates a simple fingerprint based on the Speck round function. It is meant to be used for
   creating unique-ish names for the state save file and is not intended to be cryptographically
   secure by any means. */
static uint32_t state_fingerprint(const state st) {
  assert(st.num_gates <= MAX_GATES);
  state fpstate;
  memset(&fpstate, 0, sizeof(state));
  fpstate.max_gates = st.max_gates;
  fpstate.num_gates = st.num_gates;
  for (int i = 0; i < 8; i++) {
    fpstate.outputs[i] = st.outputs[i];
  }
  for (int i = 0; i < st.num_gates; i++) {
    fpstate.gates[i].table = st.gates[i].table;
    fpstate.gates[i].type = st.gates[i].type;
    fpstate.gates[i].in1 = st.gates[i].in1;
    fpstate.gates[i].in2 = st.gates[i].in2;
    fpstate.gates[i].in3 = st.gates[i].in3;
    fpstate.gates[i].function = st.gates[i].function;
  }
  uint16_t fp1 = 0;
  uint16_t fp2 = 0;
  uint16_t *ptr = (uint16_t*)&fpstate;
  size_t len = sizeof(state) - sizeof(gate) * (MAX_GATES - fpstate.num_gates);
  for (int p = 0; p < len / 2; p++) {
    uint32_t ct = speck_round(fp1, fp2, ptr[p]);
    fp1 = ct >> 16;
    fp2 = ct & 0xffff;
  }
  if (len & 1) {
    uint32_t ct = speck_round(fp1, fp2, ((uint8_t*)&fpstate)[len - 1]);
    fp1 = ct >> 16;
    fp2 = ct & 0xffff;
  }
  for (int r = 0; r < 22; r++) {
    uint32_t ct = speck_round(fp1, fp2, 0);
    fp1 = ct >> 16;
    fp2 = ct & 0xffff;
  }
  return (((uint32_t)fp1) << 16) | fp2;
}

static void save_state(state st) {
  /* Generate a string with the output gates present in the state, in the order they were added. */
  char out[9];
  int num_outputs = 0;
  memset(out, 0, 9);
  for (int i = 0; i < st.num_gates; i++) {
    for (uint8_t k = 0; k < 8; k++) {
      if (st.outputs[k] == i) {
        num_outputs += 1;
        char str[2] = {'0' + k, '\0'};
        strcat(out, str);
        break;
      }
    }
  }

  char name[40];
  sprintf(name, "%d-%03d-%03d-%s-%08x.state", num_outputs, st.num_gates - 8, st.sat_metric, out,
      state_fingerprint(st));

  FILE *fp = fopen(name, "w");
  if (fp == NULL) {
    fprintf(stderr, "Error opening file for writing.\n");
    return;
  }
  msgpack_packer pk;
  msgpack_packer_init(&pk, fp, msgpack_fbuffer_write);
  msgpack_pack_int(&pk, MSGPACK_FORMAT_VERSION);
  msgpack_pack_int(&pk, 8); /* Number of inputs. */
  msgpack_pack_array(&pk, 8); /* Number of outputs. */
  for (int i = 0; i < 8; i++) {
    msgpack_pack_int(&pk, st.outputs[i]);
  }
  msgpack_pack_array(&pk, st.num_gates * 6);
  for (int i = 0; i < st.num_gates; i++) {
    msgpack_pack_bin(&pk, 32);
    msgpack_pack_bin_body(&pk, &st.gates[i].table, 32);
    msgpack_pack_int(&pk, st.gates[i].type);
    msgpack_pack_int(&pk, st.gates[i].in1);
    msgpack_pack_int(&pk, st.gates[i].in2);
    msgpack_pack_int(&pk, st.gates[i].in3);
    msgpack_pack_int(&pk, st.gates[i].function);
  }
  fclose(fp);
}

static int unpack_int(msgpack_unpacker *unp, int *ret) {
  assert(ret != NULL);
  if (unp == NULL) {
    return false;
  }
  msgpack_unpacked und;
  msgpack_unpacked_init(&und);
  if (msgpack_unpacker_next(unp, &und) != MSGPACK_UNPACK_SUCCESS
      || (und.data.type != MSGPACK_OBJECT_POSITIVE_INTEGER
          && und.data.type != MSGPACK_OBJECT_NEGATIVE_INTEGER)) {
    msgpack_unpacked_destroy(&und);
    return false;
  }
  *ret = und.data.via.i64;
  msgpack_unpacked_destroy(&und);
  return true;
}

/* Loads a saved state */
static bool load_state(const char *name, state *return_state) {
  assert(name != NULL);
  assert(return_state != NULL);
  FILE *fp = fopen(name, "r");
  if (fp == NULL) {
    fprintf(stderr, "Error opening file: %s\n", name);
    return false;
  }
  fseek(fp, 0, SEEK_END);
  size_t fsize = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  msgpack_unpacker unp;
  if (!msgpack_unpacker_init(&unp, fsize)) {
    return false;
  }
  if (fread(msgpack_unpacker_buffer(&unp), fsize, 1, fp) != 1) {
    return false;
  }
  fclose(fp);
  fp = NULL;
  msgpack_unpacker_buffer_consumed(&unp, fsize);

  int format_version;
  int num_inputs;
  if (!unpack_int(&unp, &format_version)
      || !unpack_int(&unp, &num_inputs)
      || format_version != MSGPACK_FORMAT_VERSION
      || num_inputs != 8) {
    msgpack_unpacker_destroy(&unp);
    return false;
  }

  msgpack_unpacked und;

  msgpack_unpacked_init(&und);
  if (msgpack_unpacker_next(&unp, &und) != MSGPACK_UNPACK_SUCCESS
      || und.data.type != MSGPACK_OBJECT_ARRAY) {
    msgpack_unpacked_destroy(&und);
    msgpack_unpacker_destroy(&unp);
    return false;
  }
  int num_outputs = und.data.via.array.size;
  if (num_outputs != 8) {
    msgpack_unpacked_destroy(&und);
    msgpack_unpacker_destroy(&unp);
    return false;
  }

  gatenum outputs[8];
  for (int i = 0; i < 8; i++) {
    if (und.data.via.array.ptr[i].type != MSGPACK_OBJECT_POSITIVE_INTEGER) {
      msgpack_unpacked_destroy(&und);
      msgpack_unpacker_destroy(&unp);
      return false;
    }
    outputs[i] = und.data.via.array.ptr[i].via.i64;
  }
  msgpack_unpacked_destroy(&und);
  msgpack_unpacked_init(&und);

  if (msgpack_unpacker_next(&unp, &und) != MSGPACK_UNPACK_SUCCESS
      || und.data.type != MSGPACK_OBJECT_ARRAY) {
    msgpack_unpacked_destroy(&und);
    msgpack_unpacker_destroy(&unp);
    return false;
  }
  int arraysize = und.data.via.array.size;

  if (arraysize % 6 != 0 || arraysize / 6 > MAX_GATES) {
    msgpack_unpacked_destroy(&und);
    msgpack_unpacker_destroy(&unp);
    return false;
  }
  for (int i = 0; i < 8; i++) {
    if (outputs[i] >= arraysize / 6 && outputs[i] != NO_GATE) {
      msgpack_unpacked_destroy(&und);
      msgpack_unpacker_destroy(&unp);
      return false;
    }
  }

  state st;
  st.max_sat_metric = INT_MAX;
  st.sat_metric = 0;
  st.max_gates = MAX_GATES;
  st.num_gates = arraysize / 6;
  memcpy(st.outputs, outputs, 8 * sizeof(gatenum));

  for (int i = 0; i < st.num_gates; i++) {
    if (und.data.via.array.ptr[i * 6].type != MSGPACK_OBJECT_BIN
        || und.data.via.array.ptr[i * 6].via.bin.size != 32
        || und.data.via.array.ptr[i * 6 + 1].type != MSGPACK_OBJECT_POSITIVE_INTEGER
        || und.data.via.array.ptr[i * 6 + 2].type != MSGPACK_OBJECT_POSITIVE_INTEGER
        || und.data.via.array.ptr[i * 6 + 3].type != MSGPACK_OBJECT_POSITIVE_INTEGER
        || und.data.via.array.ptr[i * 6 + 4].type != MSGPACK_OBJECT_POSITIVE_INTEGER
        || und.data.via.array.ptr[i * 6 + 5].type != MSGPACK_OBJECT_POSITIVE_INTEGER) {
      msgpack_unpacked_destroy(&und);
      msgpack_unpacker_destroy(&unp);
      return false;
    }
    memcpy(&st.gates[i].table, und.data.via.array.ptr[i * 6].via.bin.ptr, 32);
    st.gates[i].type = und.data.via.array.ptr[i * 6 + 1].via.i64;
    st.gates[i].in1 = und.data.via.array.ptr[i * 6 + 2].via.i64;
    st.gates[i].in2 = und.data.via.array.ptr[i * 6 + 3].via.i64;
    st.gates[i].in3 = und.data.via.array.ptr[i * 6 + 4].via.i64;
    st.gates[i].function = und.data.via.array.ptr[i * 6 + 5].via.i64;
    if (st.gates[i].type > LUT
        || st.gates[i].type < IN
        || (st.gates[i].type == IN && i >= 8)
        || (st.gates[i].type == IN && st.gates[i].in1 != NO_GATE)
        || (st.gates[i].type != IN && st.gates[i].in1 == NO_GATE)
        || ((st.gates[i].type == IN || st.gates[i].type == NOT) && st.gates[i].in2 != NO_GATE)
        || (st.gates[i].type != IN && st.gates[i].type != NOT && st.gates[i].in2 == NO_GATE)
        || (st.gates[i].type != LUT && st.gates[i].in3 != NO_GATE)
        || (st.gates[i].type == LUT && st.gates[i].in3 == NO_GATE)
        || (st.gates[i].type != LUT && st.gates[i].function != 0)
        || (st.gates[i].in1 != NO_GATE && st.gates[i].in1 >= st.num_gates)
        || (st.gates[i].in2 != NO_GATE && st.gates[i].in2 >= st.num_gates)
        || (st.gates[i].in3 != NO_GATE && st.gates[i].in3 >= st.num_gates)) {
      msgpack_unpacked_destroy(&und);
      msgpack_unpacker_destroy(&unp);
      return false;
    }
  }

  /* Calculate SAT metric. */
  for (int i = 0; i < st.num_gates; i++) {
    switch(st.gates[i].type) {
      case IN:
      case NOT:
        break;
      case AND:
      case OR:
      case ANDNOT:
        st.sat_metric += 1;
        break;
      case XOR:
        st.sat_metric += 4;
        break;
      case LUT:
        st.sat_metric = 0;
        goto no_metric;
      default:
        assert(0);
    }
  }
  no_metric:

  msgpack_unpacked_destroy(&und);
  msgpack_unpacker_destroy(&unp);
  *return_state = st;
  return true;
}

void generate_graph_one_output(const bool andnot, const bool lut, const bool randomize,
    const int iterations, const int output) {
  assert(iterations > 0);
  assert(output >= 0 && output <= 7);
  state st;
  st.max_sat_metric = INT_MAX;
  st.sat_metric = 0;
  st.max_gates = MAX_GATES;
  st.num_gates = 8;
  memset(st.gates, 0, sizeof(gate) * MAX_GATES);
  for (uint8_t i = 0; i < 8; i++) {
    st.gates[i].type = IN;
    st.gates[i].table = generate_target(i, false);
    st.gates[i].in1 = NO_GATE;
    st.gates[i].in2 = NO_GATE;
    st.gates[i].in3 = NO_GATE;
    st.gates[i].function = 0;
    st.outputs[i] = NO_GATE;
  }
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
void generate_graph(const bool andnot, const bool lut, const bool randomize, const int iterations) {
  int num_start_states = 1;
  state *start_states = malloc(sizeof(state));
  assert(start_states != NULL);
  start_states[0].max_sat_metric = INT_MAX;
  start_states[0].sat_metric = 0;
  start_states[0].max_gates = MAX_GATES;
  /* Generate the eight input bits. */
  start_states[0].num_gates = 8;
  memset(start_states[0].gates, 0, sizeof(gate) * MAX_GATES);
  for (uint8_t i = 0; i < 8; i++) {
    start_states[0].gates[i].type = IN;
    start_states[0].gates[i].table = generate_target(i, false);
    start_states[0].gates[i].in1 = NO_GATE;
    start_states[0].gates[i].in2 = NO_GATE;
    start_states[0].gates[i].in3 = NO_GATE;
    start_states[0].gates[i].function = 0;
    start_states[0].outputs[i] = NO_GATE;
  }

  /* Build the gate network one output at a time. After every added output, select the gate network
     or network with the least amount of gates and add another. */
  while (1) {
    gatenum max_gates = MAX_GATES;
    int max_sat_metric = INT_MAX;
    state *out_states = malloc(4 * sizeof(state));
    assert(out_states != NULL);
    int num_out_states = 0;
    int out_states_alloc = 4;

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
      free(out_states);
      free(start_states);
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
              if (num_out_states == out_states_alloc) {
                out_states_alloc *= 2;
                out_states = realloc(out_states, out_states_alloc * sizeof(state));
                assert(out_states != NULL);
              }
              out_states[num_out_states++] = st;
            }
          } else {
            if (max_sat_metric > st.sat_metric) {
              max_sat_metric = st.sat_metric;
              num_out_states = 0;
            }
            if (st.sat_metric <= max_sat_metric) {
              if (num_out_states == out_states_alloc) {
                out_states_alloc *= 2;
                out_states = realloc(out_states, out_states_alloc * sizeof(state));
                assert(out_states != NULL);
              }
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
    free(start_states);
    start_states = out_states;
    num_start_states = num_out_states;
  }
}

int main(int argc, char **argv) {

  bool output_dot = false;
  bool output_c = false;
  bool lut_graph = false;
  bool andnot = false;
  bool randomize = false;
  char *fname = NULL;
  int oneoutput = -1;
  int permute = 0;
  int iterations = 1;
  int c;
  while ((c = getopt(argc, argv, "c:d:hi:lno:p:rsv")) != -1) {
    switch (c) {
      case 'c':
        output_c = true;
        fname = optarg;
        break;
      case 'd':
        output_dot = true;
        fname = optarg;
        break;
      case 'h':
        printf(
            "-c file   Output C function.\n"
            "-d file   Output DOT digraph.\n"
            "-h        Display this help.\n"
            "-i n      Do n iterations per step.\n"
            "-l        Generate LUT graph.\n"
            "-n        Use ANDNOT gates.\n"
            "-o n      Generate one-output graph for output n.\n"
            "-p value  Permute sbox by XORing input with value.\n"
            "-r        Enable randomization.\n"
            "-s        Use SAT metric.\n"
            "-v        Increase verbosity.\n\n");
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
          return 1;
        }
        break;
      case 'p':
        permute = atoi(optarg);
        if (permute < 0 || permute > 255) {
          fprintf(stderr, "Bad permutation value: %s\n", optarg);
          return 1;
        }
        break;
      case 'r':
        randomize = true;
        break;
      case 's':
        g_metric = SAT;
        break;
      case 'v':
        g_verbosity += 1;
        break;
      default:
        return 1;
    }
  }

  if (output_c && output_dot) {
    fprintf(stderr, "Cannot combine c and d options.\n");
    return 1;
  }

  if (lut_graph && g_metric == SAT) {
    fprintf(stderr, "SAT metric can not be combined with LUT graph generation.\n");
    return 1;
  }

  if (output_c || output_dot) {
    state st;
    if (!load_state(fname, &st)) {
      fprintf(stderr, "Error when reading state file.\n");
      return 1;
    }
    if (output_c) {
      for (int i = 0; i < st.num_gates; i++) {
        if (st.gates[i].type == LUT) {
          fprintf(stderr, "C output of graphs containing LUTs is not supported.\n");
          return 1;
        }
      }
      print_c_function(st);
    } else {
      print_digraph(st);
    }
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

  g_rand_fp = fopen("/dev/urandom", "r");
  if (g_rand_fp == NULL) {
    fprintf(stderr, "Error opening /dev/urandom.\n");
    return 1;
  }

  if (oneoutput != -1) {
    generate_graph_one_output(andnot, lut_graph, randomize, iterations, oneoutput);
  } else {
    generate_graph(andnot, lut_graph, randomize, iterations);
  }

  assert(g_rand_fp != NULL):
  fclose(g_rand_fp);

  return 0;
}
