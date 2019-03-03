/* state.c

   Helper functions for saving and loading files containing logic circuit
   representations of S-boxes created by sboxgates.

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

#define MSGPACK_FORMAT_VERSION 2
#include <assert.h>
#include <limits.h>
#include <msgpack.h>
#include <msgpack/fbuffer.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "sboxgates.h"
#include "state.h"

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

void save_state(state st) {
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
  assert(snprintf(name, 40, "%d-%03d-%04d-%s-%08x.state", num_outputs,
    st.num_gates - get_num_inputs(&st), st.sat_metric, out, state_fingerprint(st)) < 40);

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

/* Returns the SAT metric of the specified gate type. Calling this with the LUT
 * gate type will cause an assertion to fail. */
int get_sat_metric(gate_type type) {
  switch (type) {
    case IN:
      return 0;
    case NOT:
      return 4;
    case AND:
    case OR:
    case ANDNOT:
      return 7;
    case XOR:
      return 12;
    case LUT:
    default:
      assert(0);
  }
  return 0;
}

/* Loads a saved state */
bool load_state(const char *name, state *return_state) {
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
    fclose(fp);
    return false;
  }
  if (msgpack_unpacker_buffer_capacity(&unp) < fsize) {
    if (!msgpack_unpacker_reserve_buffer(&unp, fsize)) {
      fclose(fp);
      msgpack_unpacker_destroy(&unp);
      return false;
    }
  }
  if (fread(msgpack_unpacker_buffer(&unp), fsize, 1, fp) != 1) {
    fclose(fp);
    msgpack_unpacker_destroy(&unp);
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
    if (st.gates[i].type == LUT) {
      st.sat_metric = 0;
      break;
    }
    st.sat_metric += get_sat_metric(st.gates[i].type);
  }

  msgpack_unpacked_destroy(&und);
  msgpack_unpacker_destroy(&unp);
  *return_state = st;
  return true;
}
