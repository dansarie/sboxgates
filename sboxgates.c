/*
 * Program for finding low gate count implementations of S-boxes.
 * The algorithm used is described in Kwan, Matthew: "Reducing the Gate Count of Bitslice DES."
 * IACR Cryptology ePrint Archive 2000 (2000): 51.
 *
 * Copyright (c) 2016-2017 Marcus Dansarie
 */

#include <assert.h>
#include <inttypes.h>
#ifdef _OPENMP
#include <omp.h>
#else
#warning "Compiling without OpenMP."
#endif
#include <msgpack.h>
#include <msgpack/fbuffer.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <x86intrin.h>

#define MSGPACK_FORMAT_VERSION 1
#define MSGPACK_STATE 0
#define MSGPACK_LUT_STATE 1
#define MAX_GATES 500
#define NO_GATE ((gatenum)-1)
#define PRIgatenum PRIu16

typedef enum {IN, NOT, AND, OR, XOR, ANDNOT, LUT} gate_type;

typedef __m256i ttable; /* 256 bit truth table. */
typedef uint16_t gatenum;

typedef struct {
  ttable table;
  gate_type type;
  gatenum in1; /* Input 1 to the gate. NO_GATE for the inputs. */
  gatenum in2; /* Input 2 to the gate. NO_GATE for NOT gates and the inputs. */
} gate;

typedef struct {
  gatenum max_gates;
  gatenum num_gates;  /* Current number of gates. */
  gatenum outputs[8]; /* Gate number of the respective output gates, or NO_GATE. */
  gate gates[MAX_GATES];
} state;

typedef struct {
  ttable table;
  gate_type type;
  gatenum in1; /* Input 1 to the LUT/gate, or NO_GATE. */
  gatenum in2; /* Input 2 to the LUT/gate, or NO_GATE. */
  gatenum in3; /* Input 3 to the LUT/gate, or NO_GATE. */
  uint8_t function;
} lut;

typedef struct {
  gatenum max_luts;
  gatenum num_luts;   /* Current number of LUTs. */
  gatenum outputs[8]; /* LUT number of the respective output LUTs, or NO_GATE. */
  lut luts[MAX_GATES];
} lut_state;

const uint8_t g_sbox_enc[] = {
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

ttable g_target[8];       /* Truth tables for the output bits of the sbox. */
uint8_t g_verbosity = 0;  /* Verbosity level. Higher = more debugging messages. */
bool g_andnot_available = false; /* If ANDNOT gates are available in addition to standard gates. */

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
  if (gid1 == NO_GATE || (gid2 == NO_GATE && type != NOT)) {
    return NO_GATE;
  }
  assert(type != IN);
  assert(gid1 < st->num_gates);
  assert(gid2 < st->num_gates || type == NOT);
  if (st->num_gates >= st->max_gates) {
    return NO_GATE;
  }
  st->gates[st->num_gates].type = type;
  st->gates[st->num_gates].table = table;
  st->gates[st->num_gates].in1 = gid1;
  st->gates[st->num_gates].in2 = gid2;
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
  return add_gate(st, AND, st->gates[gid1].table & st->gates[gid2].table, gid1, gid2);
}

static inline gatenum add_or_gate(state *st, gatenum gid1, gatenum gid2) {
  if (gid1 == NO_GATE || gid2 == NO_GATE) {
    return NO_GATE;
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

/* Recursively builds the gate network. The numbered comments are references to Matthew Kwan's
   paper. */
static gatenum create_circuit(state *st, const ttable target, const ttable mask,
    const int8_t *inbits) {

  /* 1. Look through the existing circuit. If there is a gate that produces the desired map, simply
     return the ID of that gate. */

  for (int i = st->num_gates - 1; i >= 0; i--) {
    if (ttable_equals_mask(target, st->gates[i].table, mask)) {
      return i;
    }
  }

  /* 2. If there are any gates whose inverse produces the desired map, append a NOT gate, and
     return the ID of the NOT gate. */

  for (int i = st->num_gates - 1; i >= 0; i--) {
    if (ttable_equals_mask(target, ~st->gates[i].table, mask)) {
      return add_not_gate(st, i);
    }
  }

  /* 3. Look at all pairs of gates in the existing circuit. If they can be combined with a single
     gate to produce the desired map, add that single gate and return its ID. */

  const ttable mtarget = target & mask;
  for (int i = st->num_gates - 1; i >= 0; i--) {
    ttable ti = st->gates[i].table & mask;
    for (int k = i - 1; k >= 0; k--) {
      ttable tk = st->gates[k].table & mask;
      if (ttable_equals(mtarget, ti | tk)) {
        return add_or_gate(st, i, k);
      }
      if (ttable_equals(mtarget, ti & tk)) {
        return add_and_gate(st, i, k);
      }
      if (ttable_equals(mtarget, ti ^ tk)) {
        return add_xor_gate(st, i, k);
      }
      if (g_andnot_available) {
        if (ttable_equals_mask(target, ~ti & tk, mask)) {
          return add_andnot_gate(st, i, k);
        }
        if (ttable_equals_mask(target, ~tk & ti, mask)) {
          return add_andnot_gate(st, k, i);
        }
      }
    }
  }

  /* 4. Look at all combinations of two or three gates in the circuit. If they can be combined with
     two gates to produce the desired map, add the gates, and return the ID of the one that produces
     the desired map. */

  for (int i = st->num_gates - 1; i >= 0; i--) {
    ttable ti = st->gates[i].table;
    for (int k = i - 1; k >= 0; k--) {
      ttable tk = st->gates[k].table;
      if (ttable_equals_mask(target, ~(ti | tk), mask)) {
        return add_nor_gate(st, i, k);
      }
      if (ttable_equals_mask(target, ~(ti & tk), mask)) {
        return add_nand_gate(st, i, k);
      }
      if (ttable_equals_mask(target, ~(ti ^ tk), mask)) {
        return add_xnor_gate(st, i, k);
      }
      if (ttable_equals_mask(target, ~ti | tk, mask)) {
        return add_or_not_gate(st, i, k);
      }
      if (ttable_equals_mask(target, ~tk | ti, mask)) {
        return add_or_not_gate(st, k, i);
      }
      if (!g_andnot_available) {
        if (ttable_equals_mask(target, ~ti & tk, mask)) {
          return add_and_not_gate(st, i, k);
        }
        if (ttable_equals_mask(target, ~tk & ti, mask)) {
          return add_and_not_gate(st, k, i);
        }
      } else if (ttable_equals_mask(target, ~ti & ~tk, mask)) {
        return add_andnot_gate(st, i, add_not_gate(st, k));
      }
    }
  }

  for (int i = st->num_gates - 1; i >= 0; i--) {
    ttable ti = st->gates[i].table & mask;
    for (int k = i - 1; k >= 0; k--) {
      ttable tk = st->gates[k].table & mask;
      ttable iandk = ti & tk;
      ttable iork = ti | tk;
      ttable ixork = ti ^ tk;
      for (int m = k - 1; m >= 0; m--) {
        ttable tm = st->gates[m].table & mask;
        if (ttable_equals(mtarget, iandk & tm)) {
          return add_and_3_gate(st, i, k, m);
        }
        if (ttable_equals(mtarget, iandk | tm)) {
          return add_and_or_gate(st, i, k, m);
        }
        if (ttable_equals(mtarget, iandk ^ tm)) {
          return add_and_xor_gate(st, i, k, m);
        }
        if (ttable_equals(mtarget, iork | tm)) {
          return add_or_3_gate(st, i, k, m);
        }
        if (ttable_equals(mtarget, iork & tm)) {
          return add_or_and_gate(st, i, k, m);
        }
        if (ttable_equals(mtarget, iork ^ tm)) {
          return add_or_xor_gate(st, i, k, m);
        }
        if (ttable_equals(mtarget, ixork ^ tm)) {
          return add_xor_3_gate(st, i, k, m);
        }
        if (ttable_equals(mtarget, ixork | tm)) {
          return add_xor_or_gate(st, i, k, m);
        }
        if (ttable_equals(mtarget, ixork & tm)) {
          return add_xor_and_gate(st, i, k, m);
        }
        ttable iandm = ti & tm;
        if (ttable_equals(mtarget, iandm | tk)) {
          return add_and_or_gate(st, i, m, k);
        }
        if (ttable_equals(mtarget, iandm ^ tk)) {
          return add_and_xor_gate(st, i, m, k);
        }
        ttable kandm = tk & tm;
        if (ttable_equals(mtarget, kandm | ti)) {
          return add_and_or_gate(st, k, m, i);
        }
        if (ttable_equals(mtarget, kandm ^ ti)) {
          return add_and_xor_gate(st, k, m, i);
        }
        ttable ixorm = ti ^ tm;
        if (ttable_equals(mtarget, ixorm | tk)) {
          return add_xor_or_gate(st, i, m, k);
        }
        if (ttable_equals(mtarget, ixorm & tk)) {
          return add_xor_and_gate(st, i, m, k);
        }
        ttable kxorm = tk ^ tm;
        if (ttable_equals(mtarget, kxorm | ti)) {
          return add_xor_or_gate(st, k, m, i);
        }
        if (ttable_equals(mtarget, kxorm & ti)) {
          return add_xor_and_gate(st, k, m, i);
        }
        ttable iorm = ti | tm;
        if (ttable_equals(mtarget, iorm & tk)) {
          return add_or_and_gate(st, i, m, k);
        }
        if (ttable_equals(mtarget, iorm ^ tk)) {
          return add_or_xor_gate(st, i, m, k);
        }
        ttable korm = tk | tm;
        if (ttable_equals(mtarget, korm & ti)) {
          return add_or_and_gate(st, k, m, i);
        }
        if (ttable_equals(mtarget, korm ^ ti)) {
          return add_or_xor_gate(st, k, m, i);
        }
        if (g_andnot_available) {
          if (ttable_equals(mtarget, ti | (~tk & tm))) {
            return add_andnot_or_gate(st, k, m, i);
          }
          if (ttable_equals(mtarget, ti | (tk & ~tm))) {
            return add_andnot_or_gate(st, m, k, i);
          }
          if (ttable_equals(mtarget, tm | (~ti & tk))) {
            return add_andnot_or_gate(st, i, k, m);
          }
          if (ttable_equals(mtarget, tm | (ti & ~tk))) {
            return add_andnot_or_gate(st, k, i, m);
          }
          if (ttable_equals(mtarget, tk | (~ti & tm))) {
            return add_andnot_or_gate(st, i, m, k);
          }
          if (ttable_equals(mtarget, tk | (ti & ~tm))) {
            return add_andnot_or_gate(st, m, i, k);
          }
          if (ttable_equals(mtarget, ~ti & (tk ^ tm))) {
            return add_xor_andnot_a_gate(st, k, m, i);
          }
          if (ttable_equals(mtarget, ~tk & (ti ^ tm))) {
            return add_xor_andnot_a_gate(st, i, m, k);
          }
          if (ttable_equals(mtarget, ~tm & (tk ^ ti))) {
            return add_xor_andnot_a_gate(st, k, i, m);
          }
          if (ttable_equals(mtarget, ti & ~(tk ^ tm))) {
            return add_xor_andnot_b_gate(st, k, m, i);
          }
          if (ttable_equals(mtarget, tk & ~(ti ^ tm))) {
            return add_xor_andnot_b_gate(st, i, m, k);
          }
          if (ttable_equals(mtarget, tm & ~(tk ^ ti))) {
            return add_xor_andnot_b_gate(st, k, i, m);
          }
          if (ttable_equals(mtarget, ~ti & tk & tm)) {
            return add_and_andnot_gate(st, i, k, m);
          }
          if (ttable_equals(mtarget, ti & ~tk & tm)) {
            return add_and_andnot_gate(st, k, i, m);
          }
          if (ttable_equals(mtarget, ti & tk & ~tm)) {
            return add_and_andnot_gate(st, m, k, i);
          }
          if (ttable_equals(mtarget, ~ti & ~tk & tm)) {
            return add_andnot_3_a_gate(st, i, k, m);
          }
          if (ttable_equals(mtarget, ~ti & tk & ~tm)) {
            return add_andnot_3_a_gate(st, i, m, k);
          }
          if (ttable_equals(mtarget, ti & ~tk & ~tm)) {
            return add_andnot_3_a_gate(st, k, m, i);
          }
          if (ttable_equals(mtarget, ti & ~(~tk & tm))) {
            return add_andnot_3_b_gate(st, k, m, i);
          }
          if (ttable_equals(mtarget, ti & ~(tk & ~tm))) {
            return add_andnot_3_b_gate(st, m, k, i);
          }
          if (ttable_equals(mtarget, tk & ~(~ti & tm))) {
            return add_andnot_3_b_gate(st, i, m, k);
          }
          if (ttable_equals(mtarget, tk & ~(ti & ~tm))) {
            return add_andnot_3_b_gate(st, m, i, k);
          }
          if (ttable_equals(mtarget, tm & ~(~tk & ti))) {
            return add_andnot_3_b_gate(st, k, i, m);
          }
          if (ttable_equals(mtarget, tm & ~(tk & ~ti))) {
            return add_andnot_3_b_gate(st, i, k, m);
          }
          if (ttable_equals(mtarget, ti ^ (~tk & tm))) {
            return add_andnot_xor_gate(st, k, m, i);
          }
          if (ttable_equals(mtarget, ti ^ (tk & ~tm))) {
            return add_andnot_xor_gate(st, m, k, i);
          }
          if (ttable_equals(mtarget, tk ^ (~ti & tm))) {
            return add_andnot_xor_gate(st, i, m, k);
          }
          if (ttable_equals(mtarget, tk ^ (ti & ~tm))) {
            return add_andnot_xor_gate(st, m, i, k);
          }
          if (ttable_equals(mtarget, tm ^ (~tk & ti))) {
            return add_andnot_xor_gate(st, k, i, m);
          }
          if (ttable_equals(mtarget, tm ^ (tk & ~ti))) {
            return add_andnot_xor_gate(st, i, k, m);
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
  assert(bitp < 6);
  next_inbits[bitp] = -1;
  next_inbits[bitp + 1] = -1;

  state best;
  best.max_gates = st->max_gates;
  best.num_gates = 0;

  /* Try all input bit orders. */
  for (int8_t bit = 0; bit < 8; bit++) {
    #ifdef _OPENMP
    #pragma omp task firstprivate(next_inbits, bitp) shared(best)
    #endif
    {
      /* Check if the current bit number has already been used for selection. */
      bool skip = false;
      for (uint8_t i = 0; i < bitp; i++) {
        if (inbits[i] == bit) {
          skip = true;
          break;
        }
      }
      if (skip) {
        goto end;
      }

      next_inbits[bitp] = bit;
      const ttable fsel = st->gates[bit].table; /* Selection bit. */

      if (g_verbosity > 1) {
        printf("Level %d: Splitting on bit %d.\n", bitp, bit);
      }
      state nst_and = *st; /* New state using AND multiplexer. */
      if (best.num_gates != 0) {
        nst_and.max_gates = best.num_gates;
      }
      gatenum fb = create_circuit(&nst_and, target & ~fsel, mask & ~fsel, next_inbits);
      gatenum mux_out_and = NO_GATE;
      if (fb != NO_GATE) {
        gatenum fc = create_circuit(&nst_and, nst_and.gates[fb].table ^ target, mask & fsel,
            next_inbits);
        gatenum andg = add_and_gate(&nst_and, fc, bit);
        mux_out_and = add_xor_gate(&nst_and, fb, andg);
      }

      state nst_or = *st; /* New state using OR multiplexer. */
      if (best.num_gates != 0) {
        nst_or.max_gates = best.num_gates;
      }
      gatenum fd = create_circuit(&nst_or, ~target & fsel, mask & fsel, next_inbits);
      gatenum mux_out_or = NO_GATE;
      if (fd != NO_GATE) {
        gatenum fe = create_circuit(&nst_or, nst_or.gates[fd].table ^ target, mask & ~fsel,
            next_inbits);
        gatenum org = add_or_gate(&nst_or, fe, bit);
        mux_out_or = add_xor_gate(&nst_or, fd, org);
      }

      if (mux_out_and == NO_GATE && mux_out_or == NO_GATE) {
        goto end;
      }
      gatenum mux_out;
      state nnst;
      if (mux_out_or == NO_GATE
          || (mux_out_and != NO_GATE && nst_and.num_gates < nst_or.num_gates)) {
        nnst = nst_and;
        mux_out = mux_out_and;
      } else {
        nnst = nst_or;
        mux_out = mux_out_or;
      }
      assert(ttable_equals_mask(target, nnst.gates[mux_out].table, mask));
      #ifdef _OPENMP
      #pragma omp critical
      #endif
      if (best.num_gates == 0 || nnst.num_gates < best.num_gates) {
        best = nnst;
        best.max_gates = st->max_gates;
      }
      /* Nasty hack to avoid using continue statement inside OpenMP task. */
      end:;
    }
  }
  #ifdef _OPENMP
  #pragma omp taskwait
  #endif
  if (best.num_gates == 0) {
    return NO_GATE;
  }
  *st = best;
  if (g_verbosity > 0) {
    printf("Level: %d Best: %d\n", bitp, best.num_gates - 1);
  }
  assert(ttable_equals_mask(target, st->gates[st->num_gates - 1].table, mask));
  return st->num_gates - 1;
}

static inline gatenum add_lut(lut_state *st, const uint8_t function, const ttable table,
    const gatenum in1, const gatenum in2, const gatenum in3) {
  if (in1 == NO_GATE || in2 == NO_GATE || in3 == NO_GATE) {
    return NO_GATE;
  }
  assert(in1 < st->num_luts);
  assert(in2 < st->num_luts);
  assert(in3 < st->num_luts);
  if (st->num_luts >= st->max_luts) {
    return NO_GATE;
  }
  st->luts[st->num_luts].type = LUT;
  st->luts[st->num_luts].function = function;
  st->luts[st->num_luts].table = table;
  st->luts[st->num_luts].in1 = in1;
  st->luts[st->num_luts].in2 = in2;
  st->luts[st->num_luts].in3 = in3;
  st->num_luts += 1;
  return st->num_luts - 1;
}

static inline gatenum add_lut_gate(lut_state *st, const gate_type type, const ttable table,
    const gatenum in1, const gatenum in2) {
  assert(type == NOT || type == AND || type == OR || type == XOR);
  assert(in1 < st->num_luts);
  assert(type == NOT || in2 < st->num_luts);
  if (st->num_luts >= st->max_luts) {
    return NO_GATE;
  }
  st->luts[st->num_luts].type = type;
  st->luts[st->num_luts].function = 0;
  st->luts[st->num_luts].table = table;
  st->luts[st->num_luts].in1 = in1;
  st->luts[st->num_luts].in2 = in2;
  st->luts[st->num_luts].in3 = NO_GATE;
  st->num_luts += 1;
  return st->num_luts - 1;
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

/* Recursively builds a network of 3-bit LUTs. */
static gatenum create_lut_circuit(lut_state *st, const ttable target, const ttable mask,
    const int8_t *inbits) {

  /* 1. Look through the existing circuit. If there is a LUT that produces the desired map, simply
     return the ID of that gate. */

  for (int i = st->num_luts - 1; i >= 0; i--) {
    if (ttable_equals_mask(target, st->luts[i].table, mask)) {
      return i;
    }
  }

  /* 2. If there are any gates whose inverse produces the desired map, append a NOT gate, and
     return the ID of the NOT gate. */

  for (int i = st->num_luts - 1; i >= 0; i--) {
    if (ttable_equals_mask(target, ~st->luts[i].table, mask)) {
      return add_lut_gate(st, NOT, ~st->luts[i].table, i, NO_GATE);
    }
  }

  /* 3. Look at all pairs of gates in the existing circuit. If they can be combined with a single
     gate to produce the desired map, add that single gate and return its ID. */

  const ttable mtarget = target & mask;
  for (int i = st->num_luts - 1; i >= 0; i--) {
    ttable ti = st->luts[i].table & mask;
    for (int k = i - 1; k >= 0; k--) {
      ttable tk = st->luts[k].table & mask;
      if (ttable_equals(mtarget, ti | tk)) {
        return add_lut_gate(st, OR, st->luts[i].table | st->luts[k].table, i, k);
      }
      if (ttable_equals(mtarget, ti & tk)) {
        return add_lut_gate(st, AND, st->luts[i].table & st->luts[k].table, i, k);
      }
      if (ttable_equals(mtarget, ti ^ tk)) {
        return add_lut_gate(st, XOR, st->luts[i].table ^ st->luts[k].table, i, k);
      }
      if (g_andnot_available) {
        if (ttable_equals(mtarget, ~ti & tk)) {
          return add_lut_gate(st, ANDNOT, ~st->luts[i].table & st->luts[k].table, i, k);
        }
        if (ttable_equals(mtarget, ~tk & ti)) {
          return add_lut_gate(st, ANDNOT, ~st->luts[k].table & st->luts[i].table, k, i);
        }
      }
    }
  }

  /* Look through all combinations of three gates in the circuit. For each combination, check if
     any of the 256 possible three bit boolean functions produces the desired map. If so, add that
     LUT and return the ID. */
  for (int i = st->num_luts - 1; i >= 0; i--) {
    const ttable ta = st->luts[i].table;
    for (int k = i - 1; k >= 0; k--) {
      const ttable tb = st->luts[k].table;
      for (int m = k - 1; m >= 0; m--) {
        const ttable tc = st->luts[m].table;
        if (!check_3lut_possible(target, mask, ta, tb, tc)) {
          continue;
        }
        uint8_t func;
        if (!get_lut_function(ta, tb, tc, target, mask, &func)) {
          continue;
        }
        ttable nt = generate_lut_ttable(func, ta, tb, tc);
        assert(ttable_equals_mask(target, nt, mask));
        return add_lut(st, func, nt, i, k, m);
      }
    }
  }

  /* Look through all combinations of five gates in the circuit. For each combination, check if
     a combination of two of the possible 256 three bit boolean functions as in LUT(LUT(a,b,c),d,e)
     produces the desired map. If so, add those LUTs and return the ID of the output LUT. */

  struct timeval before;
  if (g_verbosity > 2) {
    gettimeofday(&before, NULL);
  }

  for (int i = st->num_luts - 1; i >= 0; i--) {
    const ttable ta = st->luts[i].table;
    for (int k = i - 1; k >= 0; k--) {
      const ttable tb = st->luts[k].table;
      for (int m = k - 1; m >= 0; m--) {
        const ttable tc = st->luts[m].table;
        bool cache_set = false;
        ttable cache[256];
        for (int o = m - 1; o >= 0; o--) {
          const ttable td = st->luts[o].table;
          for (int q = o - 1; q >= 0; q--) {
            const ttable te = st->luts[q].table;
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
              return add_lut(st, func_inner, t_inner, add_lut(st, func_outer, t_outer, i, k, m),
                  o, q);
            }
          }
        }
      }
    }
  }

  if (g_verbosity > 2) {
    struct timeval after;
    gettimeofday(&after, NULL);
    double millisecs = (after.tv_sec - before.tv_sec) * 1000.0
        + (after.tv_usec - before.tv_usec) / 1000.0;
    printf("5LUT loop num luts: %" PRIgatenum " Time: %.1f ms\n", st->num_luts, millisecs);
    gettimeofday(&before, NULL);
  }

  /* Look through all combinations of seven gates in the circuit. For each combination, check if
     a combination of three of the possible 256 three bit boolean functions as in
     LUT(LUT(a,b,c),LUT(d,e,f),g) produces the desired map. If so, add those LUTs and return the ID
     of the output LUT. */

  for (int i = st->num_luts - 1; i >= 0; i--) {
    const ttable ta = st->luts[i].table;
    for (int k = i - 1; k >= 0; k--) {
      const ttable tb = st->luts[k].table;
      for (int m = k - 1; m >= 0; m--) {
        const ttable tc = st->luts[m].table;
        bool outer_cache_set = false;
        ttable outer_cache[256];
        for (int o = m - 1; o >= 0; o--) {
          const ttable td = st->luts[o].table;
          for (int q = o - 1; q >= 0; q--) {
            const ttable te = st->luts[q].table;
            for (int s = q - 1; s >= 0; s--) {
              const ttable tf = st->luts[s].table;
              bool middle_cache_set = false;
              ttable middle_cache[256];
              for (int u = s - 1; u >= 0; u--) {
                const ttable tg = st->luts[u].table;
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
                        add_lut(st, func_outer, t_outer, i, k, m),
                        add_lut(st, func_middle, t_middle, o, q, s), u);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  if (g_verbosity > 2) {
    struct timeval after;
    gettimeofday(&after, NULL);
    double millisecs = (after.tv_sec - before.tv_sec) * 1000.0
        + (after.tv_usec - before.tv_usec) / 1000.0;
    printf("7LUT loop num_luts: %" PRIgatenum " Time: %.1f ms\n", st->num_luts, millisecs);
  }

  /* Use the specified input bit to select between two Karnaugh maps. Call this function
     recursively to generate those two maps. */

  /* Copy input bits already used to new array to avoid modifying the old one. */
  int8_t next_inbits[8];
  uint8_t bitp = 0;
  while (bitp < 6 && inbits[bitp] != -1) {
    next_inbits[bitp] = inbits[bitp];
    bitp += 1;
  }
  assert(bitp < 6);
  next_inbits[bitp] = -1;
  next_inbits[bitp + 1] = -1;

  lut_state best;
  best.max_luts = st->max_luts;
  best.num_luts = 0;

  /* Try all input bit orders. */
  for (int8_t bit = 0; bit < 8; bit++) {
    #ifdef _OPENMP
    #pragma omp task firstprivate(next_inbits, bitp) shared(best)
    #endif
    {
      /* Check if the current bit number has already been used for selection. */
      bool skip = false;
      for (uint8_t i = 0; i < bitp; i++) {
        if (inbits[i] == bit) {
          skip = true;
          break;
        }
      }
      if (skip) {
        goto end;
      }

      next_inbits[bitp] = bit;
      const ttable fsel = st->luts[bit].table; /* Selection bit. */

      if (g_verbosity > 1) {
        printf("Level %d: Splitting on bit %d.\n", bitp, bit);
      }
      lut_state nnst = *st;
      #ifdef _OPENMP
      #pragma omp critical
      #endif
      if (best.num_luts != 0) {
        nnst.max_luts = best.num_luts;
      }

      gatenum fb = create_lut_circuit(&nnst, target, mask & ~fsel, next_inbits);
      if (fb == NO_GATE) {
        goto end;
      }
      gatenum fc = create_lut_circuit(&nnst, target, mask & fsel, next_inbits);
      if (fc == NO_GATE) {
        goto end;
      }
      ttable mux_table = generate_lut_ttable(0xac, nnst.luts[bit].table, nnst.luts[fb].table,
          nnst.luts[fc].table);
      gatenum out = add_lut(&nnst, 0xac, mux_table, bit, fb, fc);
      if (out == NO_GATE) {
        goto end;
      }
      assert(ttable_equals_mask(target, nnst.luts[out].table, mask));
      #ifdef _OPENMP
      #pragma omp critical
      #endif
      if (best.num_luts == 0 || nnst.num_luts < best.num_luts) {
        best = nnst;
        best.max_luts = st->max_luts;
      }
      /* Nasty hack to avoid using continue statement inside OpenMP task. */
      end:;
    }
  }
  #ifdef _OPENMP
  #pragma omp taskwait
  #endif
  if (best.num_luts == 0) {
    return NO_GATE;
  }
  *st = best;
  if (g_verbosity > 0) {
    printf("Level: %d Best: %d\n", bitp, best.num_luts - 1);
  }
  assert(ttable_equals_mask(target, st->luts[st->num_luts - 1].table, mask));
  return st->num_luts - 1;
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

/* Prints the given gate network to stdout in Graphviz dot format. */
void print_digraph(const state st) {
  printf("digraph sbox {\n");
    for (int gt = 0; gt < st.num_gates; gt++) {
      char *gatename;
      char buf[10];
      switch (st.gates[gt].type) {
        case IN:
          gatename = buf;
          sprintf(buf, "IN %d", gt);
          break;
        case NOT:
          gatename = "NOT";
          break;
        case AND:
          gatename = "AND";
          break;
        case OR:
          gatename = "OR";
          break;
        case XOR:
          gatename = "XOR";
          break;
        case ANDNOT:
          gatename = "ANDNOT";
          break;
        case LUT:
          printf("ERRLUT!\n");
        default:
          printf("%d\n", st.gates[gt].type);
          assert(false);
      }
      printf("  gt%d [label=\"%s\"];\n", gt, gatename);
    }
    for (int gt = 0; gt < st.num_gates; gt++) {
      if (st.gates[gt].in1 != NO_GATE) {
        printf("  gt%" PRIgatenum " -> gt%d;\n", st.gates[gt].in1, gt);
      }
      if (st.gates[gt].in2 != NO_GATE) {
        printf("  gt%" PRIgatenum " -> gt%d;\n", st.gates[gt].in2, gt);
      }
    }
    for (uint8_t i = 0; i < 8; i++) {
      if (st.outputs[i] != NO_GATE) {
        printf("  gt%" PRIgatenum " -> out%" PRIu8 ";\n", st.outputs[i], i);
      }
    }
  printf("}\n");
}

/* Prints the given LUT gate network to stdout in Graphviz dot format. */
void print_lut_digraph(const lut_state st) {
  printf("digraph sbox {\n");
  assert(st.num_luts < MAX_GATES);
  for (int gt = 0; gt < st.num_luts; gt++) {
    char gatename[10];
    switch (st.luts[gt].type) {
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
        sprintf(gatename, "0x%02x", st.luts[gt].function);
        break;
      default:
        assert(0);
    }
    printf("  gt%d [label=\"%s\"];\n", gt, gatename);
  }
  for (int gt = 8; gt < st.num_luts; gt++) {
    if (st.luts[gt].in1 != NO_GATE) {
      printf("  gt%" PRIgatenum " -> gt%d;\n", st.luts[gt].in1, gt);
    }
    if (st.luts[gt].in2 != NO_GATE) {
      printf("  gt%" PRIgatenum " -> gt%d;\n", st.luts[gt].in2, gt);
    }
    if (st.luts[gt].in3 != NO_GATE) {
      printf("  gt%" PRIgatenum " -> gt%d;\n", st.luts[gt].in3, gt);
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

static void save_state(const char *name, state st) {
  assert(name != NULL);
  FILE *fp = fopen(name, "w");
  if (fp == NULL) {
    fprintf(stderr, "Error opening file for writing.\n");
    return;
  }
  msgpack_packer pk;
  msgpack_packer_init(&pk, fp, msgpack_fbuffer_write);
  msgpack_pack_int(&pk, MSGPACK_FORMAT_VERSION);
  msgpack_pack_int(&pk, MSGPACK_STATE);
  msgpack_pack_int(&pk, 8); /* Number of inputs. */
  msgpack_pack_array(&pk, 8); /* Number of outputs. */
  for (int i = 0; i < 8; i++) {
    msgpack_pack_int(&pk, st.outputs[i]);
  }
  msgpack_pack_array(&pk, st.num_gates * 4);
  for (int i = 0; i < st.num_gates; i++) {
    msgpack_pack_bin(&pk, 32);
    msgpack_pack_bin_body(&pk, &st.gates[i].table, 32);
    msgpack_pack_int(&pk, st.gates[i].type);
    msgpack_pack_int(&pk, st.gates[i].in1);
    msgpack_pack_int(&pk, st.gates[i].in2);
  }
  fclose(fp);
}

static void save_lut_state(const char *name, lut_state st) {
  assert(name != NULL);
  FILE *fp = fopen(name, "w");
  if (fp == NULL) {
    fprintf(stderr, "Error opening file for writing.\n");
    return;
  }
  msgpack_packer pk;
  msgpack_packer_init(&pk, fp, msgpack_fbuffer_write);
  msgpack_pack_int(&pk, MSGPACK_FORMAT_VERSION);
  msgpack_pack_int(&pk, MSGPACK_LUT_STATE);
  msgpack_pack_int(&pk, 8); /* Number of inputs. */
  msgpack_pack_array(&pk, 8); /* Number of outputs. */
  for (int i = 0; i < 8; i++) {
    msgpack_pack_int(&pk, st.outputs[i]);
  }
  msgpack_pack_array(&pk, st.num_luts * 6);
  for (int i = 0; i < st.num_luts; i++) {
    msgpack_pack_bin(&pk, 32);
    msgpack_pack_bin_body(&pk, &st.luts[i].table, 32);
    msgpack_pack_int(&pk, st.luts[i].type);
    msgpack_pack_int(&pk, st.luts[i].in1);
    msgpack_pack_int(&pk, st.luts[i].in2);
    msgpack_pack_int(&pk, st.luts[i].in3);
    msgpack_pack_int(&pk, st.luts[i].function);
  }
  fclose(fp);
}

static int load_state(const char *name, void **return_state) {
  assert(name != NULL);
  assert(return_state != NULL);
  FILE *fp = fopen(name, "r");
  if (fp == NULL) {
    fprintf(stderr, "Error opening file: %s\n", name);
    return -1;
  }
  fseek(fp, 0, SEEK_END);
  size_t fsize = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  msgpack_unpacker unp;
  if (!msgpack_unpacker_init(&unp, fsize)) {
    return -1;
  }
  if (fread(msgpack_unpacker_buffer(&unp), fsize, 1, fp) != 1) {
    return -1;
  }
  fclose(fp);
  fp = NULL;
  msgpack_unpacker_buffer_consumed(&unp, fsize);

  msgpack_unpacked und;
  msgpack_unpacked_init(&und);

  if (msgpack_unpacker_next(&unp, &und) != MSGPACK_UNPACK_SUCCESS
      || und.data.type != MSGPACK_OBJECT_POSITIVE_INTEGER) {
    msgpack_unpacked_destroy(&und);
    msgpack_unpacker_destroy(&unp);
    return -1;
  }
  int format_version = und.data.via.i64;
  msgpack_unpacked_destroy(&und);
  if (format_version != MSGPACK_FORMAT_VERSION) {
    msgpack_unpacker_destroy(&unp);
    return -1;
  }

  if (msgpack_unpacker_next(&unp, &und) != MSGPACK_UNPACK_SUCCESS
      || und.data.type != MSGPACK_OBJECT_POSITIVE_INTEGER) {
    msgpack_unpacked_destroy(&und);
    msgpack_unpacker_destroy(&unp);
    return -1;
  }
  int state_type = und.data.via.i64;
  msgpack_unpacked_destroy(&und);
  if (state_type != MSGPACK_STATE && state_type != MSGPACK_LUT_STATE) {
    msgpack_unpacker_destroy(&unp);
    return -1;
  }

  if (msgpack_unpacker_next(&unp, &und) != MSGPACK_UNPACK_SUCCESS
      || und.data.type != MSGPACK_OBJECT_POSITIVE_INTEGER) {
    msgpack_unpacked_destroy(&und);
    msgpack_unpacker_destroy(&unp);
    return -1;
  }
  int num_inputs = und.data.via.i64;
  msgpack_unpacked_destroy(&und);
  if (num_inputs != 8) {
    msgpack_unpacker_destroy(&unp);
    return -1;
  }

  if (msgpack_unpacker_next(&unp, &und) != MSGPACK_UNPACK_SUCCESS
      || und.data.type != MSGPACK_OBJECT_ARRAY) {
    msgpack_unpacked_destroy(&und);
    msgpack_unpacker_destroy(&unp);
    return -1;
  }
  int num_outputs = und.data.via.array.size;
  if (num_outputs != 8) {
    msgpack_unpacker_destroy(&unp);
    return -1;
  }
  gatenum outputs[8];
  for (int i = 0; i < 8; i++) {
    if (und.data.via.array.ptr[i].type != MSGPACK_OBJECT_POSITIVE_INTEGER) {
      msgpack_unpacked_destroy(&und);
      msgpack_unpacker_destroy(&unp);
      return -1;
    }
    outputs[i] = und.data.via.array.ptr[i].via.i64;
  }
  msgpack_unpacked_destroy(&und);

  if (msgpack_unpacker_next(&unp, &und) != MSGPACK_UNPACK_SUCCESS
      || und.data.type != MSGPACK_OBJECT_ARRAY) {
    msgpack_unpacked_destroy(&und);
    msgpack_unpacker_destroy(&unp);
    return -1;
  }
  int arraysize = und.data.via.array.size;

  int divisor = state_type == MSGPACK_STATE ? 4 : 6;
  if (arraysize % divisor != 0 || arraysize / divisor > MAX_GATES) {
    msgpack_unpacked_destroy(&und);
    msgpack_unpacker_destroy(&unp);
    return -1;
  }
  for (int i = 0; i < 8; i++) {
    if (outputs[i] >= arraysize / divisor && outputs[i] != NO_GATE) {
      msgpack_unpacked_destroy(&und);
      msgpack_unpacker_destroy(&unp);
      return -1;
    }
  }

  if (state_type == MSGPACK_STATE) {
    state st;
    st.max_gates = MAX_GATES;
    st.num_gates = arraysize / 4;
    memcpy(st.outputs, outputs, 8 * sizeof(gatenum));
    for (int i = 0; i < st.num_gates; i++) {
      if (und.data.via.array.ptr[i * 4].type != MSGPACK_OBJECT_BIN
          || und.data.via.array.ptr[i * 4].via.bin.size != 32
          || und.data.via.array.ptr[i * 4 + 1].type != MSGPACK_OBJECT_POSITIVE_INTEGER
          || und.data.via.array.ptr[i * 4 + 2].type != MSGPACK_OBJECT_POSITIVE_INTEGER
          || und.data.via.array.ptr[i * 4 + 3].type != MSGPACK_OBJECT_POSITIVE_INTEGER) {
        msgpack_unpacked_destroy(&und);
        msgpack_unpacker_destroy(&unp);
        return -1;
      }
      memcpy(&st.gates[i].table, und.data.via.array.ptr[i * 4].via.bin.ptr, 32);
      st.gates[i].type = und.data.via.array.ptr[i * 4 + 1].via.i64;
      st.gates[i].in1 = und.data.via.array.ptr[i * 4 + 2].via.i64;
      st.gates[i].in2 = und.data.via.array.ptr[i * 4 + 3].via.i64;
      if (st.gates[i].type > ANDNOT
          || (st.gates[i].type == IN && i >= 8)
          || (st.gates[i].in1 >= st.num_gates && st.gates[i].in1 != NO_GATE)
          || (st.gates[i].in2 >= st.num_gates && st.gates[i].in2 != NO_GATE)
          || (st.gates[i].in1 == NO_GATE && i >= 8)
          || (st.gates[i].in2 == NO_GATE && i >= 8 && st.gates[i].type != NOT)) {
        msgpack_unpacked_destroy(&und);
        msgpack_unpacker_destroy(&unp);
        return -1;
      }
    }
    msgpack_unpacked_destroy(&und);
    msgpack_unpacker_destroy(&unp);
    *return_state = (void*)malloc(sizeof(state));
    if (*return_state == NULL) {
      return -1;
    }
    memcpy(*return_state, &st, sizeof(state));
    return MSGPACK_STATE;
  } else if (state_type == MSGPACK_LUT_STATE) {
    lut_state st;
    st.max_luts = MAX_GATES;
    st.num_luts = arraysize / 6;
    memcpy(st.outputs, outputs, 8 * sizeof(gatenum));
    for (int i = 0; i < st.num_luts; i++) {
      if (und.data.via.array.ptr[i * 6].type != MSGPACK_OBJECT_BIN
          || und.data.via.array.ptr[i * 6].via.bin.size != 32
          || und.data.via.array.ptr[i * 6 + 1].type != MSGPACK_OBJECT_POSITIVE_INTEGER
          || und.data.via.array.ptr[i * 6 + 2].type != MSGPACK_OBJECT_POSITIVE_INTEGER
          || und.data.via.array.ptr[i * 6 + 3].type != MSGPACK_OBJECT_POSITIVE_INTEGER
          || und.data.via.array.ptr[i * 6 + 4].type != MSGPACK_OBJECT_POSITIVE_INTEGER
          || und.data.via.array.ptr[i * 6 + 5].type != MSGPACK_OBJECT_POSITIVE_INTEGER) {
        msgpack_unpacked_destroy(&und);
        msgpack_unpacker_destroy(&unp);
        return -1;
      }
      memcpy(&st.luts[i].table, und.data.via.array.ptr[i * 6].via.bin.ptr, 32);
      st.luts[i].type = und.data.via.array.ptr[i * 6 + 1].via.i64;
      st.luts[i].in1 = und.data.via.array.ptr[i * 6 + 2].via.i64;
      st.luts[i].in2 = und.data.via.array.ptr[i * 6 + 3].via.i64;
      st.luts[i].in3 = und.data.via.array.ptr[i * 6 + 4].via.i64;
      st.luts[i].function = und.data.via.array.ptr[i * 6 + 5].via.i64;
      if (st.luts[i].type > ANDNOT
          || (st.luts[i].type == IN && i >= 8)
          || (st.luts[i].in1 >= st.num_luts && st.luts[i].in1 != NO_GATE)
          || (st.luts[i].in2 >= st.num_luts && st.luts[i].in2 != NO_GATE)
          || (st.luts[i].in3 >= st.num_luts && st.luts[i].in2 != NO_GATE)
          || (st.luts[i].in1 == NO_GATE && i >= 8)
          || (st.luts[i].in2 == NO_GATE && i >= 8 && st.luts[i].type != NOT)
          || (st.luts[i].in3 == NO_GATE && i >= 8 && st.luts[i].type == LUT)) {
        msgpack_unpacked_destroy(&und);
        msgpack_unpacker_destroy(&unp);
        return -1;
      }
    }
    msgpack_unpacked_destroy(&und);
    msgpack_unpacker_destroy(&unp);
    *return_state = (void*)malloc(sizeof(lut_state));
    if (*return_state == NULL) {
      return -1;
    }
    memcpy(*return_state, &st, sizeof(lut_state));
    return MSGPACK_LUT_STATE;
  }
  assert(0);
}

/* Called by main to generate a graph of standard (NOT, AND, OR, XOR) gates. */
void generate_gate_graph() {
  uint8_t num_start_states = 1;
  state start_states[8];
  /* Generate the eight input bits. */
  start_states[0].max_gates = MAX_GATES;
  start_states[0].num_gates = 8;
  memset(start_states[0].gates, 0, sizeof(gate) * MAX_GATES);
  for (uint8_t i = 0; i < 8; i++) {
    start_states[0].gates[i].type = IN;
    start_states[0].gates[i].table = generate_target(i, false);
    start_states[0].gates[i].in1 = NO_GATE;
    start_states[0].gates[i].in2 = NO_GATE;
    start_states[0].outputs[i] = NO_GATE;
  }

  /* Build the gate network one output at a time. After every added output, select the gate network
     or network with the least amount of gates and add another. */
  while (1) {
    gatenum max_gates = MAX_GATES;
    state out_states[8];
    uint8_t num_out_states = 0;

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
    printf("Generating circuits with %d output%s.\n", num_outputs + 1, num_outputs == 0 ? "" : "s");
    for (uint8_t current_state = 0; current_state < num_start_states; current_state++) {
      start_states[current_state].max_gates = max_gates;

      /* Add all outputs not already present to see which resulting network is the smallest. */
      for (uint8_t output = 0; output < 8; output++) {
        if (start_states[current_state].outputs[output] != NO_GATE) {
          printf("Skipping output %d.\n", output);
          continue;
        }
        printf("Generating circuit for output %d...\n", output);
        int8_t bits[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
        state st = start_states[current_state];
        st.max_gates = max_gates;
        const ttable mask = {(uint64_t)-1, (uint64_t)-1, (uint64_t)-1, (uint64_t)-1};
        #ifdef _OPENMP
        #pragma omp parallel
        #endif
        {
          #ifdef _OPENMP
          #pragma omp single
          #endif
          st.outputs[output] = create_circuit(&st, g_target[output], mask, bits);
        }
        if (st.outputs[output] == NO_GATE) {
          printf("No solution for output %d.\n", output);
          continue;
        }
        assert(ttable_equals(g_target[output], st.gates[st.outputs[output]].table));

        /* Generate a file name and save the gate network to disk. */
        char out[9];
        memset(out, 0, 9);
        for (int i = 0; i < st.num_gates; i++) {
          for (uint8_t k = 0; k < 8; k++) {
            if (st.outputs[k] == i) {
              char str[2] = {'0' + k, '\0'};
              strcat(out, str);
              break;
            }
          }
        }
        char fname[30];
        sprintf(fname, "%d-%03d-%s.state", num_outputs + 1, st.num_gates - 8, out);
        save_state(fname, st);

        if (max_gates > st.num_gates) {
          max_gates = st.num_gates;
          num_out_states = 0;
        }
        if (st.num_gates <= max_gates) {
          out_states[num_out_states++] = st;
        }
      }
    }
    printf("Found %d state%s with %d gates.\n", num_out_states,
        num_out_states == 1 ? "" : "s", max_gates - 8);
    for (uint8_t i = 0; i < num_out_states; i++) {
      start_states[i] = out_states[i];
    }
    num_start_states = num_out_states;
  }
}

/* Called by main to generate a graph of 3-bit LUTs. */
int generate_lut_graph() {
  uint8_t num_start_states = 1;
  lut_state start_states[8];

  /* Generate the eight input bits. */
  start_states[0].max_luts = MAX_GATES;
  start_states[0].num_luts = 8;
  memset(start_states[0].luts, 0, sizeof(lut) * MAX_GATES);
  for (uint8_t i = 0; i < 8; i++) {
    start_states[0].luts[i].type = IN;
    start_states[0].luts[i].function = 0;
    start_states[0].luts[i].table = generate_target(i, false);
    start_states[0].luts[i].in1 = NO_GATE;
    start_states[0].luts[i].in2 = NO_GATE;
    start_states[0].luts[i].in3 = NO_GATE;
    start_states[0].outputs[i] = NO_GATE;
  }

  /* Build the gate network one output at a time. After every added output, select the gate network
     or network with the least amount of gates and add another. */
  while (1) {
    gatenum max_luts = MAX_GATES;
    lut_state out_states[8];
    uint8_t num_out_states = 0;

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
    printf("Generating circuits with %d output%s.\n", num_outputs + 1, num_outputs == 0 ? "" : "s");
    for (uint8_t current_state = 0; current_state < num_start_states; current_state++) {
      start_states[current_state].max_luts = max_luts;

      /* Add all outputs not already present to see which resulting network is the smallest. */
      for (uint8_t output = 0; output < 8; output++) {
        if (start_states[current_state].outputs[output] != NO_GATE) {
          printf("Skipping output %d.\n", output);
          continue;
        }
        printf("Generating circuit for output %d...\n", output);
        int8_t bits[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
        lut_state st = start_states[current_state];
        st.max_luts = max_luts;
        const ttable mask = {(uint64_t)-1, (uint64_t)-1, (uint64_t)-1, (uint64_t)-1};
        #ifdef _OPENMP
        #pragma omp parallel
        #endif
        {
          #ifdef _OPENMP
          #pragma omp single
          #endif
          st.outputs[output] = create_lut_circuit(&st, g_target[output], mask, bits);
        }
        if (st.outputs[output] == NO_GATE) {
          printf("No solution for output %d.\n", output);
          continue;
        }
        assert(ttable_equals(g_target[output], st.luts[st.outputs[output]].table));

        /* Generate a file name and save the state to disk. */
        char out[9];
        memset(out, 0, 9);
        for (gatenum i = 0; i < st.num_luts; i++) {
          for (int k = 0; k < 8; k++) {
            if (st.outputs[k] == i) {
              char str[2] = {'0' + k, '\0'};
              strcat(out, str);
              break;
            }
          }
        }
        char fname[30];
        sprintf(fname, "%d-%03d-%s-lut.state", num_outputs + 1, st.num_luts - 8, out);
        save_lut_state(fname, st);

        if (max_luts > st.num_luts) {
          max_luts = st.num_luts;
          num_out_states = 0;
        }
        if (st.num_luts <= max_luts) {
          out_states[num_out_states++] = st;
        }
      }
    }
    printf("Found %d state%s with %d LUTs.\n", num_out_states,
        num_out_states == 1 ? "" : "s", max_luts - 8);
    for (int i = 0; i < num_out_states; i++) {
      start_states[i] = out_states[i];
    }
    num_start_states = num_out_states;
  }

  return 0;
}

int main(int argc, char **argv) {

  /* Generate truth tables for all output bits of the target sbox. */
  for (uint8_t i = 0; i < 8; i++) {
    g_target[i] = generate_target(i, true);
  }

  bool output_dot = false;
  bool output_c = false;
  bool lut_graph = false;
  char *fname = NULL;
  int c;
  while ((c = getopt(argc, argv, "c:d:hlnv")) != -1) {
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
            "-c file  Output C function.\n"
            "-d file  Output DOT digraph.\n"
            "-h       Display this help.\n"
            "-l       Generate LUT graph.\n"
            "-n       ANDNOT gates available.\n"
            "-v       Verbose output.\n\n");
        return 0;
      case 'l':
        lut_graph = true;
        break;
      case 'n':
        g_andnot_available = true;
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

  void *return_state = NULL;
  int loaded_state_type = -1;
  if (output_c || output_dot) {
    loaded_state_type = load_state(fname, &return_state);
    if (loaded_state_type != MSGPACK_STATE && loaded_state_type != MSGPACK_LUT_STATE) {
      fprintf(stderr, "Error when reading state file.\n");
      return 1;
    }
  }

  if (output_c) {
    assert(return_state != NULL);
    if (loaded_state_type == MSGPACK_LUT_STATE) {
      fprintf(stderr, "Outputting LUT graph as C function not supported.\n");
      return 1;
    } else if (loaded_state_type == MSGPACK_STATE) {
      print_c_function(*((state*)return_state));
      return 0;
    }
    assert(0);
  }

  if (output_dot) {
    assert(return_state != NULL);
    if (loaded_state_type == MSGPACK_LUT_STATE) {
      print_lut_digraph(*((lut_state*)return_state));
    } else if (loaded_state_type == MSGPACK_STATE) {
      print_digraph(*((state*)return_state));
    } else {
      assert(0);
    }
    return 0;
  }

  if (lut_graph) {
    printf("Generating LUT graph.\n");
    return generate_lut_graph();
  } else {
    printf("Generating standard gate graph.\n");
    generate_gate_graph();
  }

  return 0;
}
