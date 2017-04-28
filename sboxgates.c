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
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <x86intrin.h>

#ifdef USE_MPI
#include <mpi.h>
#include <sys/resource.h>

#define TAG_REQUEST 0
#define TAG_STOP 1
#define TAG_GET_AVAIL 2
#define TAG_RET_AVAIL 3
#define TAG_IS_AVAIL 4
#define TAG_START 5
#define TAG_STATE 1000
#define MIN_BITP 4
#else
#include <pthread.h>
#endif

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

#ifndef USE_MPI
typedef struct {
  ttable target;
  ttable mask;
  union {
    state *gate;
    lut_state *lut;
  } state;
  gatenum *output;
  int8_t *inbits;
  bool *done;
  bool lut;
  bool andnot_available;
  uint8_t bit;
} thread_work;
#endif

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

#ifndef USE_MPI
pthread_mutex_t g_worker_mutex;
pthread_cond_t g_worker_cond;
int g_available_workers = 0;
int g_running_threads = 0;
thread_work *g_thread_work;
bool g_stop_workers = false;
#else
int g_next_return_tag = 1000;
#endif

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

static gatenum create_circuit(state *st, const ttable target, const ttable mask,
    const int8_t *inbits, bool andnot_available);

static inline gatenum create_circuit_split(state *st, const ttable target,
    const ttable mask, const int8_t *inbits, uint8_t bit, bool andnot_available) {
  assert(st != NULL);
  assert(inbits != NULL);
  assert(bit < 8);

  const ttable fsel = st->gates[bit].table; /* Selection bit. */

  state nst_and = *st; /* New state using AND multiplexer. */
  gatenum fb = create_circuit(&nst_and, target & ~fsel, mask & ~fsel, inbits, andnot_available);
  gatenum mux_out_and = NO_GATE;
  if (fb != NO_GATE) {
    gatenum fc = create_circuit(&nst_and, nst_and.gates[fb].table ^ target, mask & fsel, inbits,
        andnot_available);
    gatenum andg = add_and_gate(&nst_and, fc, bit);
    mux_out_and = add_xor_gate(&nst_and, fb, andg);
  }

  state nst_or = *st; /* New state using OR multiplexer. */
  if (mux_out_and != NO_GATE) {
    nst_or.max_gates = nst_and.num_gates;
  }
  gatenum fd = create_circuit(&nst_or, ~target & fsel, mask & fsel, inbits, andnot_available);
  gatenum mux_out_or = NO_GATE;
  if (fd != NO_GATE) {
    gatenum fe = create_circuit(&nst_or, nst_or.gates[fd].table ^ target, mask & ~fsel, inbits,
        andnot_available);
    gatenum org = add_or_gate(&nst_or, fe, bit);
    mux_out_or = add_xor_gate(&nst_or, fd, org);
  }
  if (mux_out_and == NO_GATE && mux_out_or == NO_GATE) {
    return NO_GATE;
  }
  nst_or.max_gates = st->max_gates;
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
  *st = nnst;
  return mux_out;
}

static size_t serialize_state(state st, uint8_t **ret) {
  assert(ret != NULL);
  msgpack_sbuffer sbuf;
  msgpack_packer pk;
  msgpack_sbuffer_init(&sbuf);
  msgpack_packer_init(&pk, &sbuf, msgpack_sbuffer_write);
  msgpack_pack_int(&pk, MSGPACK_FORMAT_VERSION);
  msgpack_pack_int(&pk, MSGPACK_STATE);
  msgpack_pack_int(&pk, st.max_gates);
  msgpack_pack_int(&pk, 8); /* Number of inputs. */
  msgpack_pack_array(&pk, 8); /* Number of outputs. */
  for (int i = 0; i < 8; i++) {
    msgpack_pack_int(&pk, st.outputs[i]);
  }
  msgpack_pack_array(&pk, st.num_gates * 4);
  for (int i = 0; i < st.num_gates; i++) {
    assert(st.gates[i].type <= ANDNOT);
    msgpack_pack_bin(&pk, 32);
    msgpack_pack_bin_body(&pk, &st.gates[i].table, 32);
    msgpack_pack_int(&pk, st.gates[i].type);
    msgpack_pack_int(&pk, st.gates[i].in1);
    msgpack_pack_int(&pk, st.gates[i].in2);
  }
  *ret = (uint8_t*)malloc(sbuf.size);
  assert(*ret != NULL);
  memcpy(*ret, sbuf.data, sbuf.size);
  size_t size = sbuf.size;
  msgpack_sbuffer_destroy(&sbuf);
  return size;
}

static size_t serialize_lut_state(lut_state st, uint8_t **ret) {
  assert(ret != NULL);
  msgpack_sbuffer sbuf;
  msgpack_packer pk;
  msgpack_sbuffer_init(&sbuf);
  msgpack_packer_init(&pk, &sbuf, msgpack_sbuffer_write);
  msgpack_pack_int(&pk, MSGPACK_FORMAT_VERSION);
  msgpack_pack_int(&pk, MSGPACK_LUT_STATE);
  msgpack_pack_int(&pk, st.max_luts);
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
  *ret = (uint8_t*)malloc(sbuf.size);
  assert(*ret != NULL);
  memcpy(*ret, sbuf.data, sbuf.size);
  size_t size = sbuf.size;
  msgpack_sbuffer_destroy(&sbuf);
  return size;
}

static size_t serialize_request(uint8_t **ret, const ttable target, const ttable mask,
    const int8_t *inbits, uint8_t bit, bool andnot_available, int return_tag) {
  assert(ret != NULL);
  assert(inbits != NULL);
  msgpack_sbuffer sbuf;
  msgpack_packer pk;
  msgpack_sbuffer_init(&sbuf);
  msgpack_packer_init(&pk, &sbuf, msgpack_sbuffer_write);
  msgpack_pack_bin(&pk, 32);
  msgpack_pack_bin_body(&pk, &target, 32);
  msgpack_pack_bin(&pk, 32);
  msgpack_pack_bin_body(&pk, &mask, 32);
  msgpack_pack_array(&pk, 8);
  for (int i = 0; i < 8; i++) {
    msgpack_pack_int(&pk, inbits[i]);
  }
  msgpack_pack_int(&pk, bit);
  if (andnot_available) {
    msgpack_pack_true(&pk);
  } else {
    msgpack_pack_false(&pk);
  }
  msgpack_pack_int(&pk, return_tag);

  *ret = (uint8_t*)malloc(sbuf.size);
  assert(*ret != NULL);
  memcpy(*ret, sbuf.data, sbuf.size);
  msgpack_sbuffer_destroy(&sbuf);
  return sbuf.size;
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
static uint32_t do_fingerprint(void *st, size_t len) {
  assert(st != NULL);
  uint16_t fp1 = 0;
  uint16_t fp2 = 0;
  uint16_t *ptr = (uint16_t*)st;
  for (int p = 0; p < len / 2; p++) {
    uint32_t ct = speck_round(fp1, fp2, ptr[p]);
    fp1 = ct >> 16;
    fp2 = ct & 0xffff;
  }
  if (len % 2 != 0) {
    uint32_t ct = speck_round(fp1, fp2, ((uint8_t*)st)[len - 1]);
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

static uint32_t state_fingerprint(state st) {
  assert(st.num_gates <= MAX_GATES);
  /* Zeroize unused memory in the struct. */
  memset(st.gates + st.num_gates, 0, (MAX_GATES - st.num_gates) * sizeof(gate));
  return do_fingerprint(&st, sizeof(state));
}

static uint32_t lut_state_fingerprint(lut_state st) {
  assert(st.num_luts <= MAX_GATES);
  /* Zeroize unused memory in the struct. */
  memset(st.luts + st.num_luts, 0, (MAX_GATES - st.num_luts) * sizeof(lut));
  return do_fingerprint(&st, sizeof(lut_state));
}

#ifndef USE_MPI
static inline bool create_circuit_parallel(bool *done, gatenum *output, state *st,
    const ttable target, const ttable mask, int8_t *inbits, uint8_t bit, bool andnot_available) {
  assert(done != NULL);
  assert(output != NULL);
  assert(st != NULL);
  assert(inbits != NULL);
  assert(g_available_workers >= 0);
  *done = false;
  /* Dirty read. */
  if (g_available_workers == 0) {
    *output = create_circuit_split(st, target, mask, inbits, bit, andnot_available);
    *done = true;
    return false;
  }
  pthread_mutex_lock(&g_worker_mutex);
  if (g_available_workers > 0) {
    int threadno = 0;
    assert(g_thread_work != NULL);
    while (threadno < g_running_threads && g_thread_work[threadno].done != NULL) {
      threadno += 1;
    }
    assert(threadno != g_running_threads);
    g_available_workers -= 1;

    /* Declare a local thread_work variable and use memcpy to fill g_thread_work from it to get
       around a bug in gcc. */
    thread_work work;
    work.lut = false;
    work.done = done;
    work.state.gate = st;
    work.target = target;
    work.mask = mask;
    work.output = output;
    work.inbits = inbits;
    work.bit = bit;
    work.andnot_available = andnot_available;
    memcpy(g_thread_work + threadno, &work, sizeof(thread_work));

    pthread_cond_broadcast(&g_worker_cond);
    pthread_mutex_unlock(&g_worker_mutex);
    return true;
  } else {
    pthread_mutex_unlock(&g_worker_mutex);
    *output = create_circuit_split(st, target, mask, inbits, bit, andnot_available);
    *done = true;
    return false;
  }
}
#endif

static int deserialize_state(uint8_t *buf, size_t size, state *return_state,
    lut_state *return_lut_state) {
  assert(!(return_state == NULL && return_lut_state == NULL));
  assert(buf != NULL);
  assert(size > 0);

  msgpack_unpacker unp;
  if (!msgpack_unpacker_init(&unp, size)) {
    printf("%d\n", __LINE__);
    return -1;
  }
  if (msgpack_unpacker_buffer_capacity(&unp) < size) {
    if (!msgpack_unpacker_reserve_buffer(&unp, size)) {
      printf("%d\n", __LINE__);
      return -1;
    }
  }
  memcpy(msgpack_unpacker_buffer(&unp), buf, size);
  msgpack_unpacker_buffer_consumed(&unp, size);
  msgpack_unpacked und;
  msgpack_unpacked_init(&und);

  if (msgpack_unpacker_next(&unp, &und) != MSGPACK_UNPACK_SUCCESS
      || und.data.type != MSGPACK_OBJECT_POSITIVE_INTEGER) {
    msgpack_unpacked_destroy(&und);
    msgpack_unpacker_destroy(&unp);
    printf("%d\n", __LINE__);
    return -1;
  }
  int format_version = und.data.via.i64;
  msgpack_unpacked_destroy(&und);
  if (format_version != MSGPACK_FORMAT_VERSION) {
    msgpack_unpacker_destroy(&unp);
    printf("%d\n", __LINE__);
    return -1;
  }

  if (msgpack_unpacker_next(&unp, &und) != MSGPACK_UNPACK_SUCCESS
      || und.data.type != MSGPACK_OBJECT_POSITIVE_INTEGER) {
    msgpack_unpacked_destroy(&und);
    msgpack_unpacker_destroy(&unp);
    printf("%d\n", __LINE__);
    return -1;
  }
  int state_type = und.data.via.i64;
  msgpack_unpacked_destroy(&und);
  if (state_type != MSGPACK_STATE && state_type != MSGPACK_LUT_STATE) {
    msgpack_unpacker_destroy(&unp);
    printf("%d\n", __LINE__);
    return -1;
  }

  if (msgpack_unpacker_next(&unp, &und) != MSGPACK_UNPACK_SUCCESS
      || und.data.type != MSGPACK_OBJECT_POSITIVE_INTEGER) {
    msgpack_unpacked_destroy(&und);
    msgpack_unpacker_destroy(&unp);
    printf("%d\n", __LINE__);
    return -1;
  }
  int max_gates = und.data.via.u64;
  msgpack_unpacked_destroy(&und);
  if (state_type != MSGPACK_STATE && state_type != MSGPACK_LUT_STATE) {
    msgpack_unpacker_destroy(&unp);
    printf("%d\n", __LINE__);
    return -1;
  }

  if (msgpack_unpacker_next(&unp, &und) != MSGPACK_UNPACK_SUCCESS
      || und.data.type != MSGPACK_OBJECT_POSITIVE_INTEGER) {
    msgpack_unpacked_destroy(&und);
    msgpack_unpacker_destroy(&unp);
    printf("%d\n", __LINE__);
    return -1;
  }
  int num_inputs = und.data.via.u64;
  msgpack_unpacked_destroy(&und);
  if (num_inputs != 8) {
    msgpack_unpacker_destroy(&unp);
    printf("%d\n", __LINE__);
    return -1;
  }

  if (msgpack_unpacker_next(&unp, &und) != MSGPACK_UNPACK_SUCCESS
      || und.data.type != MSGPACK_OBJECT_ARRAY) {
    msgpack_unpacked_destroy(&und);
    msgpack_unpacker_destroy(&unp);
    printf("%d\n", __LINE__);
    return -1;
  }
  int num_outputs = und.data.via.array.size;
  if (num_outputs != 8) {
    msgpack_unpacker_destroy(&unp);
    printf("%d\n", __LINE__);
    return -1;
  }
  gatenum outputs[8];
  for (int i = 0; i < 8; i++) {
    if (und.data.via.array.ptr[i].type != MSGPACK_OBJECT_POSITIVE_INTEGER) {
      msgpack_unpacked_destroy(&und);
      msgpack_unpacker_destroy(&unp);
      printf("%d\n", __LINE__);
      return -1;
    }
    outputs[i] = und.data.via.array.ptr[i].via.u64;
  }

  if (msgpack_unpacker_next(&unp, &und) != MSGPACK_UNPACK_SUCCESS
      || und.data.type != MSGPACK_OBJECT_ARRAY) {
    msgpack_unpacked_destroy(&und);
    msgpack_unpacker_destroy(&unp);
    printf("%d\n", __LINE__);
    return -1;
  }
  int arraysize = und.data.via.array.size;

  int divisor = state_type == MSGPACK_STATE ? 4 : 6;
  if (arraysize % divisor != 0 || arraysize / divisor > MAX_GATES) {
    msgpack_unpacked_destroy(&und);
    msgpack_unpacker_destroy(&unp);
    printf("%d\n", __LINE__);
    return -1;
  }
  for (int i = 0; i < 8; i++) {
    if (outputs[i] >= arraysize / divisor && outputs[i] != NO_GATE && arraysize / divisor != 0) {
      msgpack_unpacked_destroy(&und);
      msgpack_unpacker_destroy(&unp);
      printf("%d\n", __LINE__);
      printf("i: %d outputs[i]: %d arraysize: %d divisor: %d arraysize/divisor: %d\n", i,
          outputs[i], arraysize, divisor, arraysize/divisor);
      return -1;
    }
  }

  if (state_type == MSGPACK_STATE) {
    assert(return_state != NULL);
    state st;
    memset(&st, 0, sizeof(state));
    st.max_gates = max_gates;
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
        printf("line: %d type0: %d size0: %d type1: %d type2: %d type3: %d\n", __LINE__,
            und.data.via.array.ptr[i * 4].type, und.data.via.array.ptr[i * 4].via.bin.size,
            und.data.via.array.ptr[i * 4 + 1].type, und.data.via.array.ptr[i * 4 + 2].type,
            und.data.via.array.ptr[i * 4 + 3].type);
        return -1;
      }
      memcpy(&st.gates[i].table, und.data.via.array.ptr[i * 4].via.bin.ptr, 32);
      st.gates[i].type = und.data.via.array.ptr[i * 4 + 1].via.u64;
      st.gates[i].in1 = und.data.via.array.ptr[i * 4 + 2].via.u64;
      st.gates[i].in2 = und.data.via.array.ptr[i * 4 + 3].via.u64;
      if ((st.gates[i].type != IN && st.gates[i].type != NOT && st.gates[i].type != AND
          && st.gates[i].type != OR && st.gates[i].type != XOR && st.gates[i].type != ANDNOT)
          || (st.gates[i].type == IN && i >= 8)
          || (st.gates[i].in1 >= st.num_gates && st.gates[i].in1 != NO_GATE)
          || (st.gates[i].in2 >= st.num_gates && st.gates[i].in2 != NO_GATE)
          || (st.gates[i].in1 == NO_GATE && i >= 8)
          || (st.gates[i].in2 == NO_GATE && i >= 8 && st.gates[i].type != NOT)) {
        msgpack_unpacked_destroy(&und);
        msgpack_unpacker_destroy(&unp);
        printf("line: %d num gates: %d gate num: %d gate type: %d in1: %d in2: %d\n", __LINE__,
            st.num_gates, i, st.gates[i].type, st.gates[i].in1, st.gates[i].in2);
        return -1;
      }
    }
    msgpack_unpacked_destroy(&und);
    msgpack_unpacker_destroy(&unp);
    *return_state = st;
    return MSGPACK_STATE;
  } else if (state_type == MSGPACK_LUT_STATE) {
    assert(return_lut_state != NULL);
    lut_state st;
    memset(&st, 0, sizeof(lut_state));
    st.max_luts = max_gates;
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
        printf("%d\n", __LINE__);
        return -1;
      }
      memcpy(&st.luts[i].table, und.data.via.array.ptr[i * 6].via.bin.ptr, 32);
      st.luts[i].type = und.data.via.array.ptr[i * 6 + 1].via.i64;
      st.luts[i].in1 = und.data.via.array.ptr[i * 6 + 2].via.i64;
      st.luts[i].in2 = und.data.via.array.ptr[i * 6 + 3].via.i64;
      st.luts[i].in3 = und.data.via.array.ptr[i * 6 + 4].via.i64;
      st.luts[i].function = und.data.via.array.ptr[i * 6 + 5].via.i64;
      if (st.luts[i].type > LUT
          || (st.luts[i].type == IN && i >= 8)
          || (st.luts[i].in1 >= st.num_luts && st.luts[i].in1 != NO_GATE)
          || (st.luts[i].in2 >= st.num_luts && st.luts[i].in2 != NO_GATE)
          || (st.luts[i].in3 >= st.num_luts && st.luts[i].in3 != NO_GATE)
          || (st.luts[i].in1 == NO_GATE && i >= 8)
          || (st.luts[i].in2 == NO_GATE && i >= 8 && st.luts[i].type != NOT)
          || (st.luts[i].in3 == NO_GATE && i >= 8 && st.luts[i].type == LUT)) {
        msgpack_unpacked_destroy(&und);
        msgpack_unpacker_destroy(&unp);
        printf("%d\n", __LINE__);
        printf("i: %d type: %d in1: %d in2: %d in3: %d num: %d\n", i, st.luts[i].type,
            st.luts[i].in1, st.luts[i].in2, st.luts[i].in3, st.num_luts);
        return -1;
      }
    }
    msgpack_unpacked_destroy(&und);
    msgpack_unpacker_destroy(&unp);
    *return_lut_state = st;
    return MSGPACK_LUT_STATE;
  }
  assert(0);
}

/* Recursively builds the gate network. The numbered comments are references to Matthew Kwan's
   paper. */
static gatenum create_circuit(state *st, const ttable target, const ttable mask,
    const int8_t *inbits, bool andnot_available) {

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
      if (andnot_available) {
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
      if (!andnot_available) {
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
        if (andnot_available) {
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
  int8_t next_inbits[64];
  uint8_t bitp = 0;
  while (bitp < 6 && inbits[bitp] != -1) {
    next_inbits[bitp] = inbits[bitp];
    bitp += 1;
  }
  assert(bitp < 6);
  next_inbits[bitp] = -1;
  next_inbits[bitp + 1] = -1;

  #ifdef USE_MPI
  int reserved_worker[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  uint8_t buf[800000];
  MPI_Request requests[16] = {
      MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
      MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
      MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
      MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
  const int return_tag = g_next_return_tag++;
  #else
  bool state_done[8] = {false, false, false, false, false, false, false, false};
  #endif
  bool thread_used = false;
  int num = 0;
  state new_states[8];
  gatenum output_gate[8] = {NO_GATE, NO_GATE, NO_GATE, NO_GATE, NO_GATE, NO_GATE, NO_GATE, NO_GATE};
  memset(new_states, 0, 8 * sizeof(state));

  #ifdef USE_MPI
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (bitp < MIN_BITP) {
    uint16_t reserve = 7 - bitp;
    MPI_Send(&reserve, 1, MPI_UINT16_T, 0, TAG_GET_AVAIL, MPI_COMM_WORLD);
    MPI_Recv(reserved_worker, 8, MPI_INT, 0, TAG_RET_AVAIL, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  #endif

  /* Try all input bit orders. */
  for (int bit = 0; bit < 8; bit++) {
    /* Check if the current bit number has already been used for selection. */
    bool skip = false;
    for (int i = 0; i < bitp; i++) {
      if (inbits[i] == bit) {
        skip = true;
        break;
      }
    }
    if (skip) {
      #ifndef USE_MPI
      state_done[bit] = true;
      #endif
      continue;
    }
    if (bit != 0) {
      memcpy(next_inbits + 8 * bit, next_inbits, 8 * sizeof(int8_t));
    }
    new_states[bit] = *st;
    next_inbits[8 * bit + bitp] = bit;

    #ifdef USE_MPI
    uint8_t *request = NULL;
    uint8_t *serialized_state = NULL;
    if (reserved_worker[num] > 0) {
      size_t reqsize = serialize_request(&request, target, mask, next_inbits + 8 * bit, bit,
          andnot_available, return_tag);
      size_t statesize = serialize_state(new_states[bit], &serialized_state);
      assert(reqsize > 0 && statesize > 0);
      //printf("[%4d] Sending job to [%4d].\n", rank, reserved_worker[num]);
      MPI_Send(request, reqsize, MPI_BYTE, reserved_worker[num], TAG_REQUEST, MPI_COMM_WORLD);
      MPI_Send(serialized_state, statesize, MPI_BYTE, reserved_worker[num], TAG_REQUEST,
          MPI_COMM_WORLD);
      free(request);
      free(serialized_state);
      MPI_Irecv(&output_gate[bit], 1, MPI_UINT16_T, reserved_worker[num], return_tag,
          MPI_COMM_WORLD, &requests[bit * 2]);
      MPI_Irecv(&buf[bit * 100000], 100000, MPI_BYTE, reserved_worker[num], return_tag,
          MPI_COMM_WORLD, &requests[bit * 2 + 1]);
      thread_used = true;
    } else {
      int worker = -1;
      if (bitp < MIN_BITP && num < 7 - bitp) {
        uint16_t reserve = 1;
        MPI_Send(&reserve, 1, MPI_UINT16_T, 0, TAG_GET_AVAIL, MPI_COMM_WORLD);
        MPI_Recv(&worker, 1, MPI_INT, 0, TAG_RET_AVAIL, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      if (worker > 1) {
        size_t reqsize = serialize_request(&request, target, mask, next_inbits + 8 * bit, bit,
            andnot_available, return_tag);
        size_t statesize = serialize_state(new_states[bit], &serialized_state);
        assert(reqsize > 0 && statesize > 0);
        MPI_Send(request, reqsize, MPI_BYTE, worker, TAG_REQUEST, MPI_COMM_WORLD);
        MPI_Send(serialized_state, statesize, MPI_BYTE, worker, TAG_REQUEST, MPI_COMM_WORLD);
        free(request);
        free(serialized_state);
        MPI_Irecv(&output_gate[bit], 1, MPI_UINT16_T, worker, return_tag, MPI_COMM_WORLD,
            &requests[bit * 2]);
        MPI_Irecv(&buf[bit * 100000], 100000, MPI_BYTE, worker, return_tag, MPI_COMM_WORLD,
            &requests[bit * 2 + 1]);
        thread_used = true;
      } else {
        assert(worker == -1);
        output_gate[bit] = create_circuit_split(&new_states[bit], target, mask,
            &next_inbits[8 * bit], bit, andnot_available);
      }
    }

    #else

    if (num == 7 - bitp) {
      output_gate[bit] = create_circuit_split(&new_states[bit], target, mask, &next_inbits[8 * bit],
          bit, andnot_available);
    } else {
      bool ret = create_circuit_parallel(&state_done[bit], &output_gate[bit], &new_states[bit],
          target, mask, &next_inbits[8 * bit], bit, andnot_available);
      if (ret) {
        thread_used = true;
      }
    }
    #endif

    num += 1;
  }

  #ifdef USE_MPI
  if (thread_used) {
    for (int bit = 0; bit < 8; bit++) {
      //printf("[%4d] (%d) Waiting for %d.\n", rank, bitp, 8 - bit);
      MPI_Status status1, status2;
      int size1, size2;
      MPI_Wait(&requests[bit * 2], &status1);
      MPI_Get_count(&status1, MPI_UINT16_T, &size1);
      MPI_Wait(&requests[bit * 2 + 1], &status2);
      MPI_Get_count(&status2, MPI_BYTE, &size2);
      if (size1 != 1 || output_gate[bit] == NO_GATE) {
        continue;
      }
      int ret = deserialize_state(&buf[bit * 100000], size2, &new_states[bit], NULL);
      assert(ret == MSGPACK_STATE);
      assert(ttable_equals_mask(target, new_states[bit].gates[output_gate[bit]].table, mask));
    }
  }
  #else
  bool alldone = false;
  int wait = 1;
  while (thread_used && !alldone) {
    for (int i = 0; i < 8; i++) {
      alldone = true;
      if (!state_done[i]) {
        alldone = false;
        break;
      }
    }
    if (!alldone) {
      usleep(wait);
      if (wait < 10) {
        wait *= 2;
      }
    }
  }
  #endif

  state best;
  best.num_gates = 0;
  gatenum out_gate = NO_GATE;

  for (int i = 0; i < 8; i++) {
    if (output_gate[i] == NO_GATE) {
      continue;
    }
    if (best.num_gates == 0 || best.num_gates > new_states[i].num_gates) {
      best = new_states[i];
      out_gate = output_gate[i];
    }
  }
  if (out_gate == NO_GATE) {
    return NO_GATE;
  }
  *st = best;
  if (g_verbosity > 2) {
    #ifdef USE_MPI
    printf("[%4d] Level: %d Best: %d\n", rank, bitp, best.num_gates - 8);
    #else
    printf("Level: %d Best: %d\n", bitp, best.num_gates - 8);
    #endif
  }
  assert(ttable_equals_mask(target, st->gates[out_gate].table, mask));
  return out_gate;
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

static gatenum create_lut_circuit(lut_state *st, const ttable target, const ttable mask,
    const int8_t *inbits, bool andnot_available);

static inline gatenum create_lut_circuit_split(lut_state *st, const ttable target,
    const ttable mask, const int8_t *inbits, uint8_t bit, bool andnot_available) {
  assert(st != NULL);
  assert(inbits != NULL);
  assert(bit < 8);

  const ttable fsel = st->luts[bit].table; /* Selection bit. */
  lut_state nnst = *st;

  gatenum fb = create_lut_circuit(&nnst, target, mask & ~fsel, inbits, andnot_available);
  if (fb == NO_GATE) {
    return NO_GATE;
  }

  gatenum fc = create_lut_circuit(&nnst, target, mask & fsel, inbits, andnot_available);
  if (fc == NO_GATE) {
    return NO_GATE;
  }

  ttable mux_table = generate_lut_ttable(0xac, nnst.luts[bit].table, nnst.luts[fb].table,
      nnst.luts[fc].table);
  gatenum out = add_lut(&nnst, 0xac, mux_table, bit, fb, fc);
  if (out == NO_GATE) {
    return NO_GATE;
  }
  assert(ttable_equals_mask(target, nnst.luts[out].table, mask));
  *st = nnst;
  return out;
}

#ifndef USE_MPI
static inline bool create_lut_circuit_parallel(bool *done, lut_state *st, const ttable target,
    const ttable mask, int8_t *inbits, uint8_t bit, bool andnot_available) {
  assert(done != NULL);
  assert(st != NULL);
  assert(inbits != NULL);
  assert(g_available_workers >= 0);
  *done = false;
  /* Dirty read. */
  if (g_available_workers == 0) {
    create_lut_circuit_split(st, target, mask, inbits, bit, andnot_available);
    *done = true;
    return false;
  }
  pthread_mutex_lock(&g_worker_mutex);
  if (g_available_workers > 0) {
    int threadno = 0;
    assert(g_thread_work != NULL);
    while (threadno < g_running_threads && g_thread_work[threadno].done != NULL) {
      threadno += 1;
    }
    assert(threadno != g_running_threads);
    g_available_workers -= 1;

    /* Declare a local thread_work variable and use memcpy to fill g_thread_work from it to get
       around a bug in gcc. */
    thread_work work;
    work.lut = true;
    work.done = done;
    work.state.lut = st;
    work.target = target;
    work.mask = mask;
    work.inbits = inbits;
    work.bit = bit;
    work.andnot_available = andnot_available;
    memcpy(g_thread_work + threadno, &work, sizeof(thread_work));

    pthread_cond_broadcast(&g_worker_cond);
    pthread_mutex_unlock(&g_worker_mutex);
    return true;
  } else {
    pthread_mutex_unlock(&g_worker_mutex);
    create_lut_circuit_split(st, target, mask, inbits, bit, andnot_available);
    *done = true;
    return false;
  }
}

#endif

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
    const int8_t *inbits, bool andnot_available) {

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
      if (andnot_available) {
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
  int8_t next_inbits[64];
  uint8_t bitp = 0;
  while (bitp < 6 && inbits[bitp] != -1) {
    next_inbits[bitp] = inbits[bitp];
    bitp += 1;
  }
  assert(bitp < 6);
  next_inbits[bitp] = -1;
  next_inbits[bitp + 1] = -1;

  #ifdef USE_MPI
  int reserved_worker[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  uint8_t buf[800000];
  MPI_Request requests[16] = {
      MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
      MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
      MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
      MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
  const int return_tag = g_next_return_tag++;
  #else
  bool state_done[8] = {false, false, false, false, false, false, false, false};
  #endif
  bool thread_used = false;
  int num = 0;
  lut_state new_states[8];
  gatenum output_lut[8] = {NO_GATE, NO_GATE, NO_GATE, NO_GATE, NO_GATE, NO_GATE, NO_GATE, NO_GATE};
  memset(new_states, 0, 8 * sizeof(lut_state));

  #ifdef USE_MPI
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (bitp < MIN_BITP) {
    uint16_t reserve = 7 - bitp;
    MPI_Send(&reserve, 1, MPI_UINT16_T, 0, TAG_GET_AVAIL, MPI_COMM_WORLD);
    MPI_Recv(reserved_worker, 8, MPI_INT, 0, TAG_RET_AVAIL, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  #endif

  /* Try all input bit orders. */
  for (int8_t bit = 0; bit < 8; bit++) {
    /* Check if the current bit number has already been used for selection. */
    bool skip = false;
    for (int i = 0; i < bitp; i++) {
      if (inbits[i] == bit) {
        skip = true;
        break;
      }
    }
    if (skip) {
      #ifndef USE_MPI
      state_done[bit] = true;
      #endif
      continue;
    }
    if (bit != 0) {
      memcpy(next_inbits + 8 * bit, next_inbits, 8 * sizeof(int8_t));
    }
    new_states[bit] = *st;
    next_inbits[8 * bit + bitp] = bit;
    #ifdef USE_MPI
    uint8_t *request = NULL;
    uint8_t *serialized_state = NULL;
    if (reserved_worker[num] > 0) {
      size_t reqsize = serialize_request(&request, target, mask, &next_inbits[8 * bit], bit,
          andnot_available, return_tag);
      size_t statesize = serialize_lut_state(new_states[bit], &serialized_state);
      assert(reqsize > 0 && statesize > 0);
      MPI_Send(request, reqsize, MPI_BYTE, reserved_worker[num], TAG_REQUEST, MPI_COMM_WORLD);
      MPI_Send(serialized_state, statesize, MPI_BYTE, reserved_worker[num], TAG_REQUEST,
          MPI_COMM_WORLD);
      free(request);
      free(serialized_state);
      MPI_Irecv(&output_lut[bit], 1, MPI_UINT16_T, reserved_worker[num], return_tag,
          MPI_COMM_WORLD, &requests[bit * 2]);
      MPI_Irecv(&buf[bit * 100000], 100000, MPI_BYTE, reserved_worker[num], return_tag,
          MPI_COMM_WORLD, &requests[bit * 2 + 1]);
      thread_used = true;
    } else {
      int worker = -1;
      if (bitp < MIN_BITP && num < 7 - bitp) {
        uint16_t reserve = 1;
        MPI_Send(&reserve, 1, MPI_UINT16_T, 0, TAG_GET_AVAIL, MPI_COMM_WORLD);
        MPI_Recv(&worker, 1, MPI_INT, 0, TAG_RET_AVAIL, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      if (worker > 1) {
        size_t reqsize = serialize_request(&request, target, mask, next_inbits + 8 * bit, bit,
            andnot_available, return_tag);
        size_t statesize = serialize_lut_state(new_states[bit], &serialized_state);
        assert(reqsize > 0 && statesize > 0);
        MPI_Send(request, reqsize, MPI_BYTE, worker, TAG_REQUEST, MPI_COMM_WORLD);
        MPI_Send(serialized_state, statesize, MPI_BYTE, worker, TAG_REQUEST, MPI_COMM_WORLD);
        free(request);
        free(serialized_state);
        MPI_Irecv(&output_lut[bit], 1, MPI_UINT16_T, worker, return_tag, MPI_COMM_WORLD,
            &requests[bit * 2]);
        MPI_Irecv(&buf[bit * 100000], 100000, MPI_BYTE, worker, return_tag, MPI_COMM_WORLD,
            &requests[bit * 2 + 1]);
        thread_used = true;
      } else {
        assert(worker == -1);
        output_lut[bit] = create_lut_circuit_split(&new_states[bit], target, mask,
            &next_inbits[8 * bit], bit, andnot_available);
      }
    }

    #else

    if (num == 7 - bitp) {
      output_gate[bit] = create_lut_circuit_split(&new_states[bit], target, mask,
          &next_inbits[8 * bit], bit, andnot_available);
    } else {
      bool ret = create_lut_circuit_parallel(&state_done[bit], &output_lut[bit], &new_states[bit],
          target, mask, &next_inbits[8 * bit], bit, andnot_available);
      if (ret) {
        thread_used = true;
      }
    }
    #endif

    num += 1;
  }

  #ifdef USE_MPI
  if (thread_used) {
    for (int bit = 0; bit < 8; bit++) {
      MPI_Status status1, status2;
      int size1, size2;
      MPI_Wait(&requests[bit * 2], &status1);
      MPI_Get_count(&status1, MPI_UINT16_T, &size1);
      MPI_Wait(&requests[bit * 2 + 1], &status2);
      MPI_Get_count(&status2, MPI_BYTE, &size2);
      if (size1 != 1 || output_lut[bit] == NO_GATE) {
        continue;
      }
      int ret = deserialize_state(&buf[bit * 100000], size2, NULL, &new_states[bit]);
      if (ret != MSGPACK_LUT_STATE) {
        printf("[%4d] Size: %d\n", rank, size2);
      }
      assert(ret == MSGPACK_LUT_STATE);
      assert(ttable_equals_mask(target, new_states[bit].luts[output_lut[bit]].table, mask));
    }
  }
  #else
  bool alldone = false;
  int wait = 1;
  while (thread_used && !alldone) {
    for (int i = 0; i < 8; i++) {
      alldone = true;
      if (!state_done[i]) {
        alldone = false;
        break;
      }
    }
    if (!alldone) {
      usleep(wait);
      if (wait < 10) {
        wait *= 2;
      }
    }
  }
  #endif

  lut_state best;
  best.num_luts = 0;
  gatenum out_lut = NO_GATE;

  for (int i = 0; i < 8; i++) {
    if (output_lut[i] == NO_GATE) {
      continue;
    }
    if (best.num_luts == 0 || best.num_luts > new_states[i].num_luts) {
      best = new_states[i];
      out_lut = output_lut[i];
    }
  }
  if (out_lut == NO_GATE) {
    return NO_GATE;
  }
  *st = best;
  if (g_verbosity > 2) {
    #ifdef USE_MPI
    printf("[%4d] Level: %d Best: %d\n", rank, bitp, best.num_luts - 8);
    #else
    printf("Level: %d Best: %d\n", bitp, best.num_luts - 8);
    #endif
  }
  assert(ttable_equals_mask(target, st->luts[out_lut].table, mask));
  return out_lut;
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

static bool deserialize_request(uint8_t *buf, size_t size, ttable *target, ttable *mask,
    int8_t *inbits, uint8_t *bit, bool *andnot_available, int *return_state) {
  assert(buf != NULL);
  assert(size > 0);
  assert(target != NULL);
  assert(mask != NULL);
  assert(inbits != NULL);
  assert(bit != NULL);
  assert(andnot_available != NULL);

  msgpack_unpacker unp;
  if (!msgpack_unpacker_init(&unp, size)) {
    return false;
  }
  if (msgpack_unpacker_buffer_capacity(&unp) < size) {
    if (!msgpack_unpacker_reserve_buffer(&unp, size)) {
      return false;
    }
  }
  memcpy(msgpack_unpacker_buffer(&unp), buf, size);
  msgpack_unpacker_buffer_consumed(&unp, size);
  msgpack_unpacked und;
  msgpack_unpacked_init(&und);

  if (msgpack_unpacker_next(&unp, &und) != MSGPACK_UNPACK_SUCCESS
      || und.data.type != MSGPACK_OBJECT_BIN
      || und.data.via.bin.size != 32) {
    msgpack_unpacked_destroy(&und);
    msgpack_unpacker_destroy(&unp);
    return false;
  }
  memcpy(target, und.data.via.bin.ptr, 32);

  if (msgpack_unpacker_next(&unp, &und) != MSGPACK_UNPACK_SUCCESS
      || und.data.type != MSGPACK_OBJECT_BIN
      || und.data.via.bin.size != 32) {
    msgpack_unpacked_destroy(&und);
    msgpack_unpacker_destroy(&unp);
    return false;
  }
  memcpy(mask, und.data.via.bin.ptr, 32);

  if (msgpack_unpacker_next(&unp, &und) != MSGPACK_UNPACK_SUCCESS
      || und.data.type != MSGPACK_OBJECT_ARRAY
      || und.data.via.array.size != 8) {
    msgpack_unpacked_destroy(&und);
    msgpack_unpacker_destroy(&unp);
    return false;
  }
  for (int i = 0; i < 8; i++) {
    if (und.data.via.array.ptr[i].type != MSGPACK_OBJECT_POSITIVE_INTEGER
        && und.data.via.array.ptr[i].type != MSGPACK_OBJECT_NEGATIVE_INTEGER) {
      msgpack_unpacked_destroy(&und);
      msgpack_unpacker_destroy(&unp);
      return false;
    }
    inbits[i] = und.data.via.array.ptr[i].via.i64;
  }

  if (msgpack_unpacker_next(&unp, &und) != MSGPACK_UNPACK_SUCCESS
      || und.data.type != MSGPACK_OBJECT_POSITIVE_INTEGER) {
    msgpack_unpacked_destroy(&und);
    msgpack_unpacker_destroy(&unp);
    return false;
  }
  *bit = und.data.via.u64;

  if (msgpack_unpacker_next(&unp, &und) != MSGPACK_UNPACK_SUCCESS
      || und.data.type != MSGPACK_OBJECT_BOOLEAN) {
    msgpack_unpacked_destroy(&und);
    msgpack_unpacker_destroy(&unp);
    return false;
  }
  *andnot_available = und.data.via.boolean;

  if (msgpack_unpacker_next(&unp, &und) != MSGPACK_UNPACK_SUCCESS
      || und.data.type != MSGPACK_OBJECT_POSITIVE_INTEGER) {
    msgpack_unpacked_destroy(&und);
    msgpack_unpacker_destroy(&unp);
    return false;
  }
  *return_state = und.data.via.u64;

  msgpack_unpacked_destroy(&und);
  msgpack_unpacker_destroy(&unp);
  return true;
}

static void save_state(state st) {
  uint8_t *serialized = NULL;
  size_t size = serialize_state(st, &serialized);
  assert(size > 0);
  assert(serialized != NULL);
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
  sprintf(name, "%d-%03d-%s-%08x.state", num_outputs, st.num_gates - 8, out, state_fingerprint(st));

  FILE *fp = fopen(name, "w");
  if (fp == NULL) {
    fprintf(stderr, "Error opening file for writing.\n");
    free(serialized);
    return;
  }

  if (fwrite(serialized, size, 1, fp) != 1) {
    fprintf(stderr, "Error writing to file.\n");
    fclose(fp);
    free(serialized);
    return;
  }
  free(serialized);

  fclose(fp);
}

static void save_lut_state(lut_state st) {
  uint8_t *serialized = NULL;
  size_t size = serialize_lut_state(st, &serialized);
  assert(size > 0);
  assert(serialized != NULL);

  char out[9];
  int num_outputs = 0;
  memset(out, 0, 9);
  for (int i = 0; i < st.num_luts; i++) {
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
  sprintf(name, "%d-%03d-%s-%08x-lut.state", num_outputs, st.num_luts - 8, out,
      lut_state_fingerprint(st));

  FILE *fp = fopen(name, "w");
  if (fp == NULL) {
    fprintf(stderr, "Error opening file for writing.\n");
    return;
  }

  if (fwrite(serialized, size, 1, fp) != 1) {
    fprintf(stderr, "Error writing to file.\n");
    fclose(fp);
    return;
  }
  free(serialized);
  fclose(fp);
}

static int load_state(const char *name, state *return_state, lut_state *return_lut_state) {
  assert(name != NULL);
  assert(return_state != NULL);
  assert(return_lut_state != NULL);
  FILE *fp = fopen(name, "r");
  if (fp == NULL) {
    fprintf(stderr, "Error opening file: %s\n", name);
    return -1;
  }
  fseek(fp, 0, SEEK_END);
  size_t fsize = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  uint8_t *buf = (uint8_t*)malloc(fsize);
  assert(buf != NULL);
  if (fread(buf, fsize, 1, fp) != 1) {
    free(buf);
    fclose(fp);
    return -1;
  }
  fclose(fp);
  int ret = deserialize_state(buf, fsize, return_state, return_lut_state);
  free(buf);
  return ret;
}

#ifdef USE_MPI
static gatenum mpi_create_circuit(state *st, const ttable target, const ttable mask,
    const int8_t *inbits, bool andnot_available) {
  assert(st != NULL);
  assert(inbits != NULL);
  int size;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  assert(rank == 0);
  int available[size];
  int availp = 0;

  for (int i = 2; i < size; i++) {
    available[availp++] = i;
  }

  uint8_t *request;
  uint8_t *start_state;
  int reqsize = serialize_request(&request, target, mask, inbits, 0, andnot_available, TAG_STATE);
  int statesize = serialize_state(*st, &start_state);
  assert(reqsize > 0 && statesize > 0);
  MPI_Send(request, reqsize, MPI_BYTE, 1, TAG_START, MPI_COMM_WORLD);
  MPI_Send(start_state, statesize, MPI_BYTE, 1, TAG_START, MPI_COMM_WORLD);
  free(request);
  free(start_state);
  request = NULL;
  start_state = NULL;

  uint8_t buf[100000];
  uint16_t recv;
  int serialized_size;
  int num = -1;
  MPI_Status status;
  while (true) {
    MPI_Recv(&recv, 1, MPI_UINT16_T, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    switch (status.MPI_TAG) {
      case TAG_GET_AVAIL:
        if (availp >= recv) {
          num = recv;
        } else {
          num = availp;
        }
        availp -= num;
        if (g_verbosity > 1) {
          printf("[%4d] -> [%4d] Requested %d workers. Sent %d.\n", rank, status.MPI_SOURCE, recv,
              num);
        }
        MPI_Send(&available[availp], num, MPI_INT, status.MPI_SOURCE, TAG_RET_AVAIL,
            MPI_COMM_WORLD);
        break;
      case TAG_IS_AVAIL:
        available[availp++] = status.MPI_SOURCE;
        if (g_verbosity > 1) {
          printf("[   0] <- [%4d] Available. Available workers: %d\n", status.MPI_SOURCE, availp);
        }
        break;
      case TAG_STATE:
        assert(status.MPI_SOURCE == 1);
        MPI_Recv(buf, 100000, MPI_BYTE, status.MPI_SOURCE, TAG_STATE, MPI_COMM_WORLD, &status);
        if (recv == NO_GATE) {
          return NO_GATE;
        }
        MPI_Get_count(&status, MPI_BYTE, &serialized_size);
        int ret = deserialize_state(buf, serialized_size, st, NULL);
        assert(ttable_equals(st->gates[recv].table, target));
        assert(ret == MSGPACK_STATE && st->num_gates > 7);
        return recv;
      default:
        break;
    }
  }
}

static gatenum mpi_create_lut_circuit(lut_state *st, const ttable target, const ttable mask,
    const int8_t *inbits, bool andnot_available) {
  assert(st != NULL);
  assert(inbits != NULL);
  int size;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  assert(rank == 0);
  int available[size];
  int availp = 0;

  for (int i = 2; i < size; i++) {
    available[availp++] = i;
  }

  uint8_t *request;
  uint8_t *start_state;
  int reqsize = serialize_request(&request, target, mask, inbits, 0, andnot_available, TAG_STATE);
  int statesize = serialize_lut_state(*st, &start_state);
  assert(reqsize > 0 && statesize > 0);
  MPI_Send(request, reqsize, MPI_BYTE, 1, TAG_START, MPI_COMM_WORLD);
  MPI_Send(start_state, statesize, MPI_BYTE, 1, TAG_START, MPI_COMM_WORLD);
  free(request);
  free(start_state);
  request = NULL;
  start_state = NULL;

  uint8_t buf[100000];
  uint16_t recv;
  int serialized_size;
  int num = -1;
  MPI_Status status;
  while (true) {
    MPI_Recv(&recv, 1, MPI_UINT16_T, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    switch (status.MPI_TAG) {
      case TAG_GET_AVAIL:
        if (availp >= recv) {
          num = recv;
        } else {
          num = availp;
        }
        availp -= num;
        if (g_verbosity > 1) {
          printf("[%4d] -> [%4d] Requested %d workers. Sent %d.\n", rank, status.MPI_SOURCE, recv,
              num);
        }
        MPI_Send(&available[availp], num, MPI_INT, status.MPI_SOURCE, TAG_RET_AVAIL,
            MPI_COMM_WORLD);
        break;
      case TAG_IS_AVAIL:
        available[availp++] = status.MPI_SOURCE;
        if (g_verbosity > 1) {
          printf("[   0] <- [%4d] Available. Available workers: %d\n", status.MPI_SOURCE, availp);
        }
        break;
      case TAG_STATE:
        assert(status.MPI_SOURCE == 1);
        MPI_Recv(buf, 100000, MPI_BYTE, status.MPI_SOURCE, TAG_STATE, MPI_COMM_WORLD,
            &status);
        if (recv == NO_GATE) {
          return NO_GATE;
        }
        MPI_Get_count(&status, MPI_BYTE, &serialized_size);
        int ret = deserialize_state(buf, serialized_size, NULL, st);
        assert(ttable_equals(st->luts[recv].table, target));
        assert(ret == MSGPACK_LUT_STATE && st->num_luts > 7);
        return recv;
      default:
        break;
    }
  }
}


#endif

/* Called by main to generate a graph of standard (NOT, AND, OR, XOR) gates. */
void generate_gate_graph(bool andnot_available) {
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
        #ifdef USE_MPI
        st.outputs[output] = mpi_create_circuit(&st, g_target[output], mask, bits,
            andnot_available);
        #else
        st.outputs[output] = create_circuit(&st, g_target[output], mask, bits, andnot_available);
        #endif
        if (st.outputs[output] == NO_GATE) {
          printf("No solution for output %d.\n", output);
          continue;
        } else {
          printf("Solution for output %d: %d gates. Fingerprint: %08x\n", output, st.num_gates - 8,
              state_fingerprint(st));
        }
        assert(ttable_equals(g_target[output], st.gates[st.outputs[output]].table));
        save_state(st);

        if (max_gates > st.num_gates) {
          max_gates = st.num_gates;
          out_states[0] = st;
          num_out_states = 1;
        } else if (max_gates == st.num_gates) {
          out_states[num_out_states++] = st;
        }
      }
    }
    printf("Found %d state%s with %d gates.\n", num_out_states, num_out_states == 1 ? "" : "s",
        max_gates - 8);
    for (uint8_t i = 0; i < num_out_states; i++) {
      start_states[i] = out_states[i];
    }
    num_start_states = num_out_states;
  }
}

/* Called by main to generate a graph of 3-bit LUTs. */
void generate_lut_graph(bool andnot_available) {
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
        #ifdef USE_MPI
        st.outputs[output] = mpi_create_lut_circuit(&st, g_target[output], mask, bits,
            andnot_available);
        #else
        st.outputs[output] = create_lut_circuit(&st, g_target[output], mask, bits,
            andnot_available);
        #endif
        if (st.outputs[output] == NO_GATE) {
          printf("No solution for output %d.\n", output);
          continue;
        } else {
          printf("Solution for output %d: %d LUTs. Fingerprint: %08x\n", output, st.num_luts - 8,
              lut_state_fingerprint(st));
        }
        assert(ttable_equals(g_target[output], st.luts[st.outputs[output]].table));
        save_lut_state(st);

        if (max_luts > st.num_luts) {
          max_luts = st.num_luts;
          out_states[0] = st;
          num_out_states = 1;
        } else if (max_luts == st.num_luts) {
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
}

#ifdef USE_MPI
void mpi_worker() {
  bool stop = false;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  uint8_t buf[100000];
  uint8_t buf2[100000];
  int reqsize;
  while (!stop) {
    MPI_Status status;
    MPI_Recv(buf, 100000, MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    switch (status.MPI_TAG) {
      case TAG_REQUEST:
      case TAG_START:
        //printf("[%4d] Got job from [%4d].\n", rank, status.MPI_SOURCE);
        MPI_Get_count(&status, MPI_BYTE, &reqsize);
        MPI_Status status2;
        MPI_Recv(buf2, 100000, MPI_BYTE, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD,
            &status2);
        int statesize;
        MPI_Get_count(&status2, MPI_BYTE, &statesize);
        ttable target;
        ttable mask;
        int8_t inbits[8];
        uint8_t bit;
        bool andnot_available;
        state st;
        lut_state lst;
        int return_tag;
        bool deserialize_ret = deserialize_request(buf, reqsize, &target, &mask, inbits, &bit,
            &andnot_available, &return_tag);
        assert(deserialize_ret);
        int ret = deserialize_state(buf2, statesize, &st, &lst);
        assert(ret == MSGPACK_STATE || ret == MSGPACK_LUT_STATE);
        uint8_t *serialized = NULL;
        gatenum outgate;
        if (ret == MSGPACK_STATE) {
          if (status.MPI_TAG == TAG_REQUEST) {
            outgate = create_circuit_split(&st, target, mask, inbits, bit, andnot_available);
          } else {
            assert(status.MPI_TAG == TAG_START && status.MPI_SOURCE == 0);
            outgate = create_circuit(&st, target, mask, inbits, andnot_available);
          }
          MPI_Send(&outgate, 1, MPI_UINT16_T, status.MPI_SOURCE, return_tag, MPI_COMM_WORLD);
          if (outgate != NO_GATE) {
            assert(ttable_equals_mask(target, st.gates[outgate].table, mask));
            int retsize = serialize_state(st, &serialized);
            assert(retsize > 0);
            MPI_Send(serialized, retsize, MPI_BYTE, status.MPI_SOURCE, return_tag, MPI_COMM_WORLD);
            free(serialized);
            serialized = NULL;
          } else {
            MPI_Send(NULL, 0, MPI_BYTE, status.MPI_SOURCE, return_tag, MPI_COMM_WORLD);
          }
        } else {
          if (status.MPI_TAG == TAG_REQUEST) {
            outgate = create_lut_circuit_split(&lst, target, mask, inbits, bit, andnot_available);
          } else {
            assert(status.MPI_TAG == TAG_START && status.MPI_SOURCE == 0);
            outgate = create_lut_circuit(&lst, target, mask, inbits, andnot_available);
          }
          MPI_Send(&outgate, 1, MPI_UINT16_T, status.MPI_SOURCE, return_tag, MPI_COMM_WORLD);
          if (outgate != NO_GATE) {
            assert(ttable_equals_mask(target, lst.luts[outgate].table, mask));
            int retsize = serialize_lut_state(lst, &serialized);
            assert(retsize > 0);
            MPI_Send(serialized, retsize, MPI_BYTE, status.MPI_SOURCE, return_tag, MPI_COMM_WORLD);
            free(serialized);
            serialized = NULL;
          } else {
            MPI_Send(NULL, 0, MPI_BYTE, status.MPI_SOURCE, return_tag, MPI_COMM_WORLD);
          }
        }
        uint16_t msg = 0;
        if (status.MPI_TAG != TAG_START) {
          MPI_Send(&msg, 1, MPI_UINT16_T, 0, TAG_IS_AVAIL, MPI_COMM_WORLD);
        }
        break;
      case TAG_STOP:
        stop = true;
        break;
      default:
        fprintf(stderr, "[%4d] Received unexpected tag: %d\n", rank, status.MPI_TAG);
        break;
    }
  }
}

#else
void *worker_thread(void *arg) {
  pthread_mutex_lock(&g_worker_mutex);
  int threadno = g_running_threads++;
  g_available_workers += 1;
  while (!g_stop_workers) {
    while (g_thread_work[threadno].done == NULL) {
      pthread_cond_wait(&g_worker_cond, &g_worker_mutex);
      if (g_stop_workers) {
        pthread_mutex_unlock(&g_worker_mutex);
        return NULL;
      }
    }
    /* Declare a local thread_work variable and use memcpy to fill it with data from g_thread_work
       to get around a bug in gcc. */
    thread_work work;
    memcpy(&work, g_thread_work + threadno, sizeof(thread_work));
    if (g_verbosity > 1) {
      printf("[%d] Starting work. Bit: %d\n", threadno, work.bit);
    }
    pthread_mutex_unlock(&g_worker_mutex);
    if (work.lut) {
      create_lut_circuit_split(work.state.lut, work.target, work.mask, work.inbits,
          work.bit, work.andnot_available);

    } else {
      *work.output = create_circuit_split(work.state.gate, work.target, work.mask, work.inbits,
          work.bit, work.andnot_available);
    }
    if (g_verbosity > 1) {
      printf("[%d] Done.\n", threadno);
    }
    pthread_mutex_lock(&g_worker_mutex);
    *work.done = true;
    g_available_workers += 1;
    memset(g_thread_work + threadno, 0, sizeof(thread_work));
  }
  g_running_threads -= 1;
  g_available_workers -= 1;
  pthread_mutex_unlock(&g_worker_mutex);
  return NULL;
}
#endif

int main(int argc, char **argv) {

  #ifdef USE_MPI
  MPI_Init(&argc, &argv);
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    fprintf(stderr, "At least two MPI nodes are needed to run.\n");
    MPI_Finalize();
    return 1;
  }
  if (rank == 0) {
    printf("Size: %d\n", size);
  }
  struct rlimit rl;
  getrlimit(RLIMIT_NOFILE, &rl);
  rl.rlim_cur = rl.rlim_max;
  setrlimit(RLIMIT_NOFILE, &rl);
  #endif

  /* Generate truth tables for all output bits of the target sbox. */
  for (uint8_t i = 0; i < 8; i++) {
    g_target[i] = generate_target(i, true);
  }

  bool output_dot = false;
  bool output_c = false;
  bool lut_graph = false;
  bool andnot_available = false;
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
        #ifdef USE_MPI
        MPI_Finalize();
        #endif
        return 0;
      case 'l':
        lut_graph = true;
        break;
      case 'n':
        andnot_available = true;
        break;
      case 'v':
        g_verbosity += 1;
        break;
      default:
        #ifdef USE_MPI
        MPI_Finalize();
        #endif
        return 1;
    }
  }

  if (output_c && output_dot) {
    fprintf(stderr, "Cannot combine c and d options.\n");
    #ifdef USE_MPI
    MPI_Finalize();
    #endif
    return 1;
  }

  state return_state;
  lut_state return_lut_state;
  int loaded_state_type = -1;
  if (output_c || output_dot) {
    #ifdef USE_MPI
    if (rank != 0) {
      MPI_Finalize();
      return 0;
    }
    #endif
    loaded_state_type = load_state(fname, &return_state, &return_lut_state);
    if (loaded_state_type != MSGPACK_STATE && loaded_state_type != MSGPACK_LUT_STATE) {
      fprintf(stderr, "Error when reading state file.\n");
      #ifdef USE_MPI
      MPI_Finalize();
      #endif
      return 1;
    }
  }

  if (output_c) {
    #ifdef USE_MPI
    if (rank != 0) {
      MPI_Finalize();
      return 0;
    }
    #endif
    if (loaded_state_type == MSGPACK_LUT_STATE) {
      fprintf(stderr, "Outputting LUT graph as C function not supported.\n");
      #ifdef USE_MPI
      MPI_Finalize();
      #endif
      return 1;
    } else if (loaded_state_type == MSGPACK_STATE) {
      print_c_function(return_state);
      #ifdef USE_MPI
      MPI_Finalize();
      #endif
      return 0;
    }
    assert(0);
  }

  if (output_dot) {
    #ifdef USE_MPI
    if (rank != 0) {
      MPI_Finalize();
      return 0;
    }
    #endif
    if (loaded_state_type == MSGPACK_LUT_STATE) {
      print_lut_digraph(return_lut_state);
    } else if (loaded_state_type == MSGPACK_STATE) {
      print_digraph(return_state);
    } else {
      assert(0);
    }
    #ifdef USE_MPI
    MPI_Finalize();
    #endif
    return 0;
  }

  #ifdef USE_MPI
  if (rank == 0) {
    if (lut_graph) {
      printf("Generating LUT graph.\n");
      generate_lut_graph(andnot_available);
    } else {
      printf("Generating standard gate graph.\n");
      generate_gate_graph(andnot_available);
    }
    uint8_t msg = 0;
    for (int i = 1; i < size; i++) {
      MPI_Send(&msg, 1, MPI_BYTE, i, TAG_STOP, MPI_COMM_WORLD);
    }
  } else {
    mpi_worker();
  }

  MPI_Finalize();
  #else
  int numproc = sysconf(_SC_NPROCESSORS_ONLN) - 1;
  pthread_mutex_init(&g_worker_mutex, NULL);
  pthread_cond_init(&g_worker_cond, NULL);
  g_thread_work = (thread_work*)calloc(numproc, sizeof(thread_work));
  assert(g_thread_work != NULL);
  pthread_t thread[numproc];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, 0x800000);
  for (int i = 0; i < numproc; i++) {
    pthread_create(thread + i, &attr, worker_thread, NULL);
  }
  pthread_attr_destroy(&attr);

  if (lut_graph) {
    printf("Generating LUT graph.\n");
    generate_lut_graph(andnot_available);
  } else {
    printf("Generating standard gate graph.\n");
    generate_gate_graph(andnot_available);
  }

  g_stop_workers = true;
  pthread_cond_broadcast(&g_worker_cond);

  for (int i = 0; i < numproc; i++) {
    void *ptr;
    pthread_join(thread[i], &ptr);
  }

  free(g_thread_work);
  pthread_cond_destroy(&g_worker_cond);
  pthread_mutex_destroy(&g_worker_mutex);
  #endif

  return 0;
}
