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
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <x86intrin.h>

#define MAX_GATES 500
#define NO_GATE ((uint64_t)-1)

typedef enum {IN, NOT, AND, OR, XOR} gate_type;

typedef __m256i ttable; /* 256 bit truth table. */

typedef struct {
  gate_type type;
  ttable table;
  uint64_t in1; /* Input 1 to the gate. NO_GATE for the inputs. */
  uint64_t in2; /* Input 2 to the gate. NO_GATE for NOT gates and the inputs. */
} gate;

typedef struct {
  uint64_t max_gates;
  uint64_t num_gates;  /* Current number of gates. */
  uint64_t outputs[8]; /* Gate number of the respective output gates, or NO_GATE. */
  gate gates[MAX_GATES];
} state;

typedef struct {
  uint8_t function;
  ttable table;
  uint64_t in1; /* Input 1 to the LUT. NO_GATE for the inputs. */
  uint64_t in2; /* Input 2 to the LUT. NO_GATE for the inputs. */
  uint64_t in3; /* Input 3 to the LUT. NO_GATE for the inputs. */
} lut;

typedef struct {
  uint64_t max_luts;
  uint64_t num_luts;   /* Current number of LUTs. */
  uint64_t outputs[8]; /* LUT number of the respective output LUTs, or NO_GATE. */
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
uint8_t *g_lutlut = NULL; /* Lookup table used to calculate truth tables for LUTs faster. */

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
static inline uint64_t add_gate(state *st, gate_type type, ttable table, uint64_t gid1,
    uint64_t gid2) {
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

static inline uint64_t add_not_gate(state *st, uint64_t gid) {
  return add_gate(st, NOT, ~st->gates[gid].table, gid, NO_GATE);
}

static inline uint64_t add_and_gate(state *st, uint64_t gid1, uint64_t gid2) {
  return add_gate(st, AND, st->gates[gid1].table & st->gates[gid2].table, gid1, gid2);
}

static inline uint64_t add_or_gate(state *st, uint64_t gid1, uint64_t gid2) {
  return add_gate(st, OR, st->gates[gid1].table | st->gates[gid2].table, gid1, gid2);
}

static inline uint64_t add_xor_gate(state *st, uint64_t gid1, uint64_t gid2) {
  return add_gate(st, XOR, st->gates[gid1].table ^ st->gates[gid2].table, gid1, gid2);
}

static inline uint64_t add_nand_gate(state *st, uint64_t gid1, uint64_t gid2) {
  return add_not_gate(st, add_and_gate(st, gid1, gid2));
}

static inline uint64_t add_nor_gate(state *st, uint64_t gid1, uint64_t gid2) {
  return add_not_gate(st, add_or_gate(st, gid1, gid2));
}

static inline uint64_t add_xnor_gate(state *st, uint64_t gid1, uint64_t gid2) {
  return add_not_gate(st, add_xor_gate(st, gid1, gid2));
}

static inline uint64_t add_or_not_gate(state *st, uint64_t gid1, uint64_t gid2) {
  return add_or_gate(st, add_not_gate(st, gid1), gid2);
}

static inline uint64_t add_and_not_gate(state *st, uint64_t gid1, uint64_t gid2) {
  return add_and_gate(st, add_not_gate(st, gid1), gid2);
}

static inline uint64_t add_or_3_gate(state *st, uint64_t gid1, uint64_t gid2, uint64_t gid3) {
  return add_or_gate(st, add_or_gate(st, gid1, gid2), gid3);
}

static inline uint64_t add_and_3_gate(state *st, uint64_t gid1, uint64_t gid2, uint64_t gid3) {
  return add_and_gate(st, add_and_gate(st, gid1, gid2), gid3);
}

static inline uint64_t add_xor_3_gate(state *st, uint64_t gid1, uint64_t gid2, uint64_t gid3) {
  return add_xor_gate(st, add_xor_gate(st, gid1, gid2), gid3);
}

static inline uint64_t add_and_or_gate(state *st, uint64_t gid1, uint64_t gid2, uint64_t gid3) {
  return add_or_gate(st, add_and_gate(st, gid1, gid2), gid3);
}

static inline uint64_t add_and_xor_gate(state *st, uint64_t gid1, uint64_t gid2, uint64_t gid3) {
  return add_xor_gate(st, add_and_gate(st, gid1, gid2), gid3);
}

static inline uint64_t add_xor_or_gate(state *st, uint64_t gid1, uint64_t gid2, uint64_t gid3) {
  return add_or_gate(st, add_xor_gate(st, gid1, gid2), gid3);
}

static inline uint64_t add_xor_and_gate(state *st, uint64_t gid1, uint64_t gid2, uint64_t gid3) {
  return add_and_gate(st, add_xor_gate(st, gid1, gid2), gid3);
}

static inline uint64_t add_or_and_gate(state *st, uint64_t gid1, uint64_t gid2, uint64_t gid3) {
  return add_and_gate(st, add_or_gate(st, gid1, gid2), gid3);
}

static inline uint64_t add_or_xor_gate(state *st, uint64_t gid1, uint64_t gid2, uint64_t gid3) {
  return add_xor_gate(st, add_or_gate(st, gid1, gid2), gid3);
}

/* Recursively builds the gate network. The numbered comments are references to Matthew Kwan's
   paper. */
static uint64_t create_circuit(state *st, const ttable target, const ttable mask,
    const int8_t *inbits) {

  /* 1. Look through the existing circuit. If there is a gate that produces the desired map, simply
     return the ID of that gate. */

  for (int64_t i = st->num_gates - 1; i >= 0; i--) {
    if (ttable_equals_mask(target, st->gates[i].table, mask)) {
      return i;
    }
  }

  /* 2. If there are any gates whose inverse produces the desired map, append a NOT gate, and
     return the ID of the NOT gate. */

  for (int64_t i = st->num_gates - 1; i >= 0; i--) {
    if (ttable_equals_mask(target, ~st->gates[i].table, mask)) {
      return add_not_gate(st, i);
    }
  }

  /* 3. Look at all pairs of gates in the existing circuit. If they can be combined with a single
     gate to produce the desired map, add that single gate and return its ID. */

  const ttable mtarget = target & mask;
  for (int64_t i = st->num_gates - 1; i >= 0; i--) {
    ttable ti = st->gates[i].table & mask;
    for (int64_t k = i - 1; k >= 0; k--) {
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
    }
  }

  /* 4. Look at all combinations of two or three gates in the circuit. If they can be combined with
     two gates to produce the desired map, add the gates, and return the ID of the one that produces
     the desired map. */

  for (int64_t i = st->num_gates - 1; i >= 0; i--) {
    ttable ti = st->gates[i].table;
    for (int64_t k = i - 1; k >= 0; k--) {
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
      if (ttable_equals_mask(target, ~ti & tk, mask)) {
        return add_and_not_gate(st, i, k);
      }
      if (ttable_equals_mask(target, ~tk & ti, mask)) {
        return add_and_not_gate(st, k, i);
      }
    }
  }

  for (int64_t i = st->num_gates - 1; i >= 0; i--) {
    ttable ti = st->gates[i].table & mask;
    for (int64_t k = i - 1; k >= 0; k--) {
      ttable tk = st->gates[k].table & mask;
      ttable iandk = ti & tk;
      ttable iork = ti | tk;
      ttable ixork = ti ^ tk;
      for (int64_t m = k - 1; m >= 0; m--) {
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
#ifdef _OPENMP
#pragma omp parallel for firstprivate(next_inbits, bitp)
#endif
  for (int8_t bit = 0; bit < 8; bit++) {
    /* Check if the current bit number has already been used for selection. */
    bool skip = false;
    for (uint8_t i = 0; i < bitp; i++) {
      if (inbits[i] == bit) {
        skip = true;
        break;
      }
    }
    if (skip) {
      continue;
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
    uint64_t fb = create_circuit(&nst_and, target & ~fsel, mask & ~fsel, next_inbits);
    uint64_t mux_out_and = NO_GATE;
    if (fb != NO_GATE) {
      uint64_t fc = create_circuit(&nst_and, nst_and.gates[fb].table ^ target, mask & fsel,
          next_inbits);
      uint64_t andg = add_and_gate(&nst_and, fc, bit);
      mux_out_and = add_xor_gate(&nst_and, fb, andg);
    }

    state nst_or = *st; /* New state using OR multiplexer. */
    if (best.num_gates != 0) {
      nst_or.max_gates = best.num_gates;
    }
    uint64_t fd = create_circuit(&nst_or, ~target & fsel, mask & fsel, next_inbits);
    uint64_t mux_out_or = NO_GATE;
    if (fd != NO_GATE) {
      uint64_t fe = create_circuit(&nst_or, nst_or.gates[fd].table ^ target, mask & ~fsel,
          next_inbits);
      uint64_t org = add_or_gate(&nst_or, fe, bit);
      mux_out_or = add_xor_gate(&nst_or, fd, org);
    }

    if (mux_out_and == NO_GATE && mux_out_or == NO_GATE) {
      continue;
    }
    uint64_t mux_out;
    state nnst;
    if (mux_out_or == NO_GATE || (mux_out_and != NO_GATE && nst_and.num_gates < nst_or.num_gates)) {
      nnst = nst_and;
      mux_out = mux_out_and;
    } else {
      nnst = nst_or;
      mux_out = mux_out_or;
    }
    assert(ttable_equals_mask(target, nnst.gates[mux_out].table, mask));
    if (best.num_gates == 0 || nnst.num_gates < best.num_gates) {
      best = nnst;
      best.max_gates = st->max_gates;
    }
  }
  if (best.num_gates == 0) {
    return NO_GATE;
  }
  *st = best;
  if (g_verbosity > 0) {
    printf("Level: %d Best: %" PRIu64 "\n", bitp, best.num_gates - 1);
  }
  assert(ttable_equals_mask(target, st->gates[st->num_gates - 1].table, mask));
  return st->num_gates - 1;
}

static inline ttable generate_lut_ttable(const uint8_t function, const ttable in1, const ttable in2,
    const ttable in3) {
  ttable out;
  uint8_t *outp = (uint8_t*)(&out);
  uint8_t *in1p = (uint8_t*)(&in1);
  uint8_t *in2p = (uint8_t*)(&in2);
  uint8_t *in3p = (uint8_t*)(&in3);
  for (uint8_t i = 0; i < 32; i++) {
    uint32_t addr1 = (function << 12) | ((in1p[i] & 0xf) << 8) | ((in2p[i] & 0xf) << 4)
        | (in3p[i] & 0xf);
    uint32_t addr2 = (function << 12) | ((in1p[i] & 0xf0) << 4) | (in2p[i] & 0xf0)
        | ((in3p[i] & 0xf0) >> 4);
    outp[i] = g_lutlut[addr1] | (g_lutlut[addr2] << 4);
  }
  return out;
}

static inline uint64_t add_lut(lut_state *st, const uint8_t function, const ttable table,
    const uint64_t in1, const uint64_t in2, const uint64_t in3) {
  if (in1 == NO_GATE || in2 == NO_GATE || in3 == NO_GATE) {
    return NO_GATE;
  }
  assert(in1 < st->num_luts);
  assert(in2 < st->num_luts);
  assert(in3 < st->num_luts);
  if (st->num_luts >= st->max_luts) {
    return NO_GATE;
  }
  st->luts[st->num_luts].function = function;
  st->luts[st->num_luts].table = table;
  st->luts[st->num_luts].in1 = in1;
  st->luts[st->num_luts].in2 = in2;
  st->luts[st->num_luts].in3 = in3;
  st->num_luts += 1;
  return st->num_luts - 1;
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
        ttable k = tt1 & tt2 & tt3;
        if (ttable_equals_mask(target & k, k, mask)) {
          match |= k;
        } else if (!_mm256_testz_si256(target & k & mask, target & k & mask)) {
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

/* Recursively builds a network of 3-bit LUTs. */
static uint64_t create_lut_circuit(lut_state *st, const ttable target, const ttable mask,
    const int8_t *inbits) {

  /* Look through the existing circuit. If there is a LUT that produces the desired map, simply
     return the ID of that gate. */

  for (int64_t i = st->num_luts - 1; i >= 0; i--) {
    if (ttable_equals_mask(target, st->luts[i].table, mask)) {
      return i;
    }
  }

  /* Look through all combinations of three gates in the circuit. For each combination, check if
     any of the 256 possible three bit boolean functions produces the desired map. If so, add that
     LUT and return the ID. */
  for (int64_t i = st->num_luts - 1; i >= 0; i--) {
    const ttable ta = st->luts[i].table;
    for (int64_t k = i - 1; k >= 0; k--) {
      const ttable tb = st->luts[k].table;
      for (int64_t m = k - 1; m >= 0; m--) {
        const ttable tc = st->luts[m].table;
        if (!check_3lut_possible(target, mask, ta, tb, tc)) {
          continue;
        }
        uint8_t func = 0;
        do {
          ttable nt = generate_lut_ttable(func, ta, tb, tc);
          if (ttable_equals_mask(target, nt, mask)) {
            return add_lut(st, func, nt, i, k, m);
          }
          func += 1;
        } while (func != 0);
      }
    }
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
#ifdef _OPENMP
#pragma omp parallel for firstprivate(next_inbits, bitp)
#endif
  for (int8_t bit = 0; bit < 8; bit++) {
    /* Check if the current bit number has already been used for selection. */
    bool skip = false;
    for (uint8_t i = 0; i < bitp; i++) {
      if (inbits[i] == bit) {
        skip = true;
        break;
      }
    }
    if (skip) {
      continue;
    }

    next_inbits[bitp] = bit;
    const ttable fsel = st->luts[bit].table; /* Selection bit. */

    if (g_verbosity > 1) {
      printf("Level %d: Splitting on bit %d.\n", bitp, bit);
    }
    lut_state nnst = *st;
    if (best.num_luts != 0) {
      nnst.max_luts = best.num_luts;
    }
    uint64_t fb = create_lut_circuit(&nnst, target, mask & ~fsel, next_inbits);
    if (fb == NO_GATE) {
      continue;
    }
    uint64_t fc = create_lut_circuit(&nnst, target, mask & fsel, next_inbits);
    if (fc == NO_GATE) {
      continue;
    }
    ttable mux_table = generate_lut_ttable(0xac, nnst.luts[bit].table, nnst.luts[fb].table,
        nnst.luts[fc].table);
    uint64_t out = add_lut(&nnst, 0xac, mux_table, bit, fb, fc);
    if (out == NO_GATE) {
      continue;
    }
    assert(ttable_equals_mask(target, nnst.luts[out].table, mask));
    if (best.num_luts == 0 || nnst.num_luts < best.num_luts) {
      best = nnst;
      best.max_luts = st->max_luts;
    }
  }
  if (best.num_luts == 0) {
    return NO_GATE;
  }
  *st = best;
  if (g_verbosity > 0) {
    printf("Level: %d Best: %" PRIu64 "\n", bitp, best.num_luts - 1);
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
    for (uint64_t gt = 0; gt < st.num_gates; gt++) {
      char *gatename;
      char buf[10];
      switch (st.gates[gt].type) {
        case IN:
          gatename = buf;
          sprintf(buf, "IN %" PRIu64, gt);
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
        default:
          assert(false);
      }
      printf("  gt%" PRIu64 " [label=\"%s\"];\n", gt, gatename);
    }
    for (uint64_t gt = 0; gt < st.num_gates; gt++) {
      if (st.gates[gt].in1 != NO_GATE) {
        printf("  gt%" PRIu64 " -> gt%" PRIu64 ";\n", st.gates[gt].in1, gt);
      }
      if (st.gates[gt].in2 != NO_GATE) {
        printf("  gt%" PRIu64 " -> gt%" PRIu64 ";\n", st.gates[gt].in2, gt);
      }
    }
    for (uint8_t i = 0; i < 8; i++) {
      if (st.outputs[i] != NO_GATE) {
        printf("  gt%" PRIu64 " -> out%" PRIu8 ";\n", st.outputs[i], i);
      }
    }
  printf("}\n");
}

/* Prints the given LUT gate network to stdout in Graphviz dot format. */
void print_lut_digraph(const lut_state st) {
  printf("digraph sbox {\n");
  assert(st.num_luts < MAX_GATES);
  for (uint64_t gt = 0; gt < st.num_luts; gt++) {
    char gatename[10];
    if (st.luts[gt].in1 == NO_GATE) {
      sprintf(gatename, "IN %" PRIu64, gt);
    } else {
      sprintf(gatename, "0x%02x", st.luts[gt].function);
    }
    printf("  gt%" PRIu64 " [label=\"%s\"];\n", gt, gatename);
  }
  for (uint64_t gt = 8; gt < st.num_luts; gt++) {
    printf("  gt%" PRIu64 " -> gt%" PRIu64 ";\n", st.luts[gt].in1, gt);
    printf("  gt%" PRIu64 " -> gt%" PRIu64 ";\n", st.luts[gt].in2, gt);
    printf("  gt%" PRIu64 " -> gt%" PRIu64 ";\n", st.luts[gt].in3, gt);
  }
  for (uint8_t i = 0; i < 8; i++) {
    if (st.outputs[i] != NO_GATE) {
      printf("  gt%" PRIu64 " -> out%" PRIu8 ";\n", st.outputs[i], i);
    }
  }
  printf("}\n");
}

/* Called by print_c_function to get variable names. */
static bool get_c_variable_name(const state st, const uint64_t gate, char *buf) {
  if (gate < 8) {
    sprintf(buf, "in.b%" PRIu64, gate);
    return false;
  }
  for (uint8_t i = 0; i < 8; i++) {
    if (st.outputs[i] == gate) {
      sprintf(buf, "out%d", i);
      return false;
    }
  }
  sprintf(buf, "var%" PRIu64, gate);
  return true;
}

/* Converts the given state gate network to a C function and prints it to stdout. */
static void print_c_function(const state st) {
  const char TYPE[] = "int ";
  char buf[10];
  printf("__device__ __forceinline__ int s0(eightbits in) {\n");
  for (uint64_t gate = 8; gate < st.num_gates; gate++) {
    bool ret = get_c_variable_name(st, gate, buf);
    printf("  %s%s = ", ret == true ? TYPE : "", buf);
    get_c_variable_name(st, st.gates[gate].in1, buf);
    if (st.gates[gate].type == NOT) {
      printf("~%s;\n", buf);
      continue;
    }
    printf("%s ", buf);
    switch (st.gates[gate].type) {
      case AND:
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

/* Saves a state struct to file. */
static void save_state(const char *name, void *st, bool lut) {
  FILE *fp = fopen(name, "w");
  if (fp == NULL) {
    fprintf(stderr, "Error opening file for writing.\n");
    return;
  }
  size_t size = lut ? sizeof(lut_state) : sizeof(state);
  if (fwrite(st, size, 1, fp) != 1) {
    fprintf(stderr, "File write error.\n");
  }
  fclose(fp);
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
    uint64_t max_gates = MAX_GATES;
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
        st.outputs[output] = create_circuit(&st, g_target[output], mask, bits);
        if (st.outputs[output] == NO_GATE) {
          printf("No solution for output %d.\n", output);
          continue;
        }
        assert(ttable_equals(g_target[output], st.gates[st.outputs[output]].table));

        /* Generate a file name and save the gate network to disk. */
        char out[9];
        memset(out, 0, 9);
        for (uint64_t i = 0; i < st.num_gates; i++) {
          for (uint8_t k = 0; k < 8; k++) {
            if (st.outputs[k] == i) {
              char str[2] = {'0' + k, '\0'};
              strcat(out, str);
              break;
            }
          }
        }
        char fname[30];
        sprintf(fname, "%d-%03" PRIu64 "-%s.state", num_outputs + 1, st.num_gates - 8, out);
        save_state(fname, &st, false);

        if (max_gates > st.num_gates) {
          max_gates = st.num_gates;
          num_out_states = 0;
        }
        if (st.num_gates <= max_gates) {
          out_states[num_out_states++] = st;
        }
      }
    }
    printf("Found %d state%s with %" PRIu64 " gates.\n", num_out_states,
        num_out_states == 1 ? "" : "s", max_gates);
    for (uint8_t i = 0; i < num_out_states; i++) {
      start_states[i] = out_states[i];
    }
    num_start_states = num_out_states;
  }
}

/* Called by main to generate a graph of 3-bit LUTs. */
int generate_lut_graph() {
  printf("Creating lookup tables...\n");
  assert(g_lutlut == NULL);
  g_lutlut = (uint8_t*)malloc(0x100000 * sizeof(uint8_t));
  if (g_lutlut == NULL) {
    fprintf(stderr, "Could not allocate 1 MB of memory for the lookup table.\n");
    return 1;
  }
  for (uint32_t func = 0; func < 256; func++) {
    for (uint32_t i1 = 0; i1 < 16; i1++) {
      for (uint32_t i2 = 0; i2 < 16; i2++) {
        for (uint32_t i3 = 0; i3 < 16; i3++) {
          uint8_t val = 0;
          for (uint8_t bit = 0; bit < 4; bit++) {
            val = val << 1;
            uint8_t sel = ((i1 >> (3 - bit)) & 1) << 2;
            sel |= ((i2 >> (3 - bit)) & 1) << 1;
            sel |= (i3 >> (3 - bit)) & 1;
            val |= (func >> sel) & 1;
          }
          uint32_t addr = (func << 12) | (i1 << 8) | (i2 << 4) | i3;
          g_lutlut[addr] = val;
        }
      }
    }
  }

  uint8_t num_start_states = 1;
  lut_state start_states[8];

  /* Generate the eight input bits. */
  start_states[0].max_luts = MAX_GATES;
  start_states[0].num_luts = 8;
  memset(start_states[0].luts, 0, sizeof(lut) * MAX_GATES);
  for (uint8_t i = 0; i < 8; i++) {
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
    uint64_t max_luts = MAX_GATES;
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
        st.outputs[output] = create_lut_circuit(&st, g_target[output], mask, bits);
        if (st.outputs[output] == NO_GATE) {
          printf("No solution for output %d.\n", output);
          continue;
        }
        assert(ttable_equals(g_target[output], st.luts[st.outputs[output]].table));

        /* Generate a file name and save the state to disk. */
        char out[9];
        memset(out, 0, 9);
        for (uint64_t i = 0; i < st.num_luts; i++) {
          for (uint8_t k = 0; k < 8; k++) {
            if (st.outputs[k] == i) {
              char str[2] = {'0' + k, '\0'};
              strcat(out, str);
              break;
            }
          }
        }
        char fname[30];
        sprintf(fname, "%d-%03" PRIu64 "-%s-lut.state", num_outputs + 1, st.num_luts - 8, out);
        save_state(fname, &st, true);

        if (max_luts > st.num_luts) {
          max_luts = st.num_luts;
          num_out_states = 0;
        }
        if (st.num_luts <= max_luts) {
          out_states[num_out_states++] = st;
        }
      }
    }
    printf("Found %d state%s with %" PRIu64 " LUTs.\n", num_out_states,
        num_out_states == 1 ? "" : "s", max_luts - 8);
    for (uint8_t i = 0; i < num_out_states; i++) {
      start_states[i] = out_states[i];
    }
    num_start_states = num_out_states;
  }

  free(g_lutlut);
  g_lutlut = NULL;

  return 0;
}

int main(int argc, char **argv) {

  /* Generate truth tables for all output bits of the target sbox. */
  for (uint8_t i = 0; i < 8; i++) {
    g_target[i] = generate_target(i, true);
  }

  state loaded_state;
  lut_state lut_loaded_state;

  bool output_dot = false;
  bool output_c = false;
  bool lut_graph = false;
  char *fname;
  int c;
  while ((c = getopt(argc, argv, "c:d:hlv")) != -1) {
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
            "-v       Verbose output.\n\n");
        return 0;
      case 'l':
        lut_graph = true;
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

  if (output_c || output_dot) {
    FILE *fp = fopen(fname, "r");
    if (fp == NULL) {
      fprintf(stderr, "Error opening file: %s\n", fname);
      return 1;
    }
    int read = 0;
    if (lut_graph) {
      read = fread(&lut_loaded_state, sizeof(lut_state), 1, fp);
    } else {
      read = fread(&loaded_state, sizeof(state), 1, fp);
    }
    if (read != 1) {
      fprintf(stderr, "Error reading file: %s\n", fname);
      fclose(fp);
      return 1;
    }
    fclose(fp);
  }

  if (output_c) {
    if (lut_graph) {
      fprintf(stderr, "Outputting LUT graph as C function not supported.\n");
      return 1;
    }
    print_c_function(loaded_state);
    return 0;
  }

  if (output_dot) {
    if (lut_graph) {
      print_lut_digraph(lut_loaded_state);
    } else {
      print_digraph(loaded_state);
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
