/* state.c

   Helper functions for saving and loading files containing logic circuit
   representations of S-boxes created by sboxgates.

   Copyright (c) 2016-2017, 2020-2021 Marcus Dansarie

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
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "lut.h"
#include "sboxgates.h"
#include "state.h"

const char* const gate_name[] = {
  "FALSE",
  "AND",
  "A_AND_NOT_B",
  "A",
  "NOT_A_AND_B",
  "B",
  "XOR",
  "OR",
  "NOR",
  "XNOR",
  "NOT_B",
  "A_OR_NOT_B",
  "NOT_A",
  "NOT_A_OR_B",
  "NAND",
  "TRUE",
  "NOT",
  "IN",
  "LUT"
};

/* The Speck round function. */
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
  assert(snprintf(name, 40, "%d-%03d-%04d-%s-%08x.xml", num_outputs,
    st.num_gates - get_num_inputs(&st), st.sat_metric, out, state_fingerprint(st)) < 40);

  FILE *fp = fopen(name, "w");
  if (fp == NULL) {
    fprintf(stderr, "Error opening file for writing. (state.c:%d)\n", __LINE__);
    return;
  }

  fprintf(fp, "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n");
  fprintf(fp, "<gates>\n");
  for (int i = 0; i < 8; i++) {
    if (st.outputs[i] != NO_GATE) {
      fprintf(fp, "  <output bit=\"%d\" gate=\"%d\" />\n", i, st.outputs[i]);
    }
  }
  for (int i = 0; i < st.num_gates; i++) {
    const char *type = NULL;
    assert(st.gates[i].type <= LUT);
    type = gate_name[st.gates[i].type];
    if (st.gates[i].type == IN) {
      fprintf(fp, "  <gate type=\"IN\" />\n");
    } else {
      if (st.gates[i].type == LUT) {
        fprintf(fp, "  <gate type=\"LUT\" function=\"%02x\">\n", st.gates[i].function);
      } else {
        fprintf(fp, "  <gate type=\"%s\">\n", type);
      }
      if (st.gates[i].in1 != NO_GATE) {
        fprintf(fp, "    <input gate=\"%d\" />\n", st.gates[i].in1);
      }
      if (st.gates[i].in2 != NO_GATE) {
        fprintf(fp, "    <input gate=\"%d\" />\n", st.gates[i].in2);
      }
      if (st.gates[i].in3 != NO_GATE) {
        fprintf(fp, "    <input gate=\"%d\" />\n", st.gates[i].in3);
      }
      fprintf(fp, "  </gate>\n");
    }
  }
  fprintf(fp, "</gates>\n");
  fclose(fp);
}

int get_sat_metric(gate_type type) {
  switch (type) {
    case FALSE_GATE:  return 1;
    case AND:         return 7;
    case A_AND_NOT_B: return 4;
    case A:           return 4;
    case NOT_A_AND_B: return 7;
    case B:           return 4;
    case XOR:         return 12;
    case OR:          return 7;
    case NOR:         return 7;
    case XNOR:        return 12;
    case NOT_B:       return 4;
    case A_OR_NOT_B:  return 7;
    case NOT_A:       return 4;
    case NOT_A_OR_B:  return 7;
    case NAND:        return 7;
    case TRUE_GATE:   return 1;
    case NOT:         return 4;
    case IN:          return 0;
    case LUT:
    default:          assert(0);
  }
}

int get_num_inputs(const state *st) {
  int inputs = 0;
  for (int i = 0; st->gates[i].type == IN && i < st->num_gates; i++) {
    inputs += 1;
  }
  return inputs;
}

/* Calculates the truth table of a LUT given its function and three input truth tables. */
ttable generate_lut_ttable(const uint8_t function, const ttable in1, const ttable in2,
    const ttable in3) {
  ttable ret = {0};
  if (function & 1) {
    ret |= ~in1 & ~in2 & ~in3;
  }
  if (function & 2) {
    ret |= ~in1 & ~in2 & in3;
  }
  if (function & 4) {
    ret |= ~in1 & in2 & ~in3;
  }
  if (function & 8) {
    ret |= ~in1 & in2 & in3;
  }
  if (function & 16) {
    ret |= in1 & ~in2 & ~in3;
  }
  if (function & 32) {
    ret |= in1 & ~in2 & in3;
  }
  if (function & 64) {
    ret |= in1 & in2 & ~in3;
  }
  if (function & 128) {
    ret |= in1 & in2 & in3;
  }
  return ret;
}

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

#define LOAD_STATE_RETURN_ON_ERROR(X, Y)\
  if (X) {\
    fprintf(stderr, "Error when parsing XML document. (state.c:%d)\n", __LINE__);\
    if (Y != NULL) xmlFreeDoc(Y);\
    return false;\
  }

/* Loads a saved state */
bool load_state(const char *name, state *return_state) {
  assert(name != NULL);
  assert(return_state != NULL);

  xmlDocPtr doc = xmlParseFile(name);
  LOAD_STATE_RETURN_ON_ERROR(doc == NULL, doc);

  /* Get gates. */
  xmlNodePtr gates = NULL;
  for (xmlNodePtr ptr = doc->children; ptr != NULL; ptr = ptr->next) {
    if (strcmp((char*)ptr->name, "gates") == 0) {
      gates = ptr;
      break;
    }
  }
  LOAD_STATE_RETURN_ON_ERROR(gates == NULL, doc);

  state st;
  memset(&st, 0, sizeof(state));
  st.max_gates = MAX_GATES;
  for (int i = 0; i < 8; i++) {
    st.outputs[i] = NO_GATE;
  }

  /* Parse gates. */
  for (xmlNodePtr gate = gates->children; gate != NULL; gate = gate->next) {
    if (strcmp((char*)gate->name, "gate") != 0) {
      continue;
    }

    /* Parse type enum. */
    char *typestr = (char*)xmlGetProp(gate, (xmlChar*)"type");
    LOAD_STATE_RETURN_ON_ERROR(typestr == NULL, doc);
    gate_type type = 0;
    while (type <= LUT) {
      if (strcmp(typestr, gate_name[type]) == 0) {
        break;
      }
      type += 1;
    }
    xmlFree(typestr);
    if (type > LUT) {
      LOAD_STATE_RETURN_ON_ERROR(true, doc);
    }
    typestr = NULL;

    /* Parse LUT function. */
    long func = 0;
    char *funcstr = (char*)xmlGetProp(gate, (xmlChar*)"function");
    if (funcstr != NULL) {
      func = strtol(funcstr, NULL, 16);
      xmlFree(funcstr);
      funcstr = NULL;
      LOAD_STATE_RETURN_ON_ERROR(func <= 0 || func > 255, doc);
    }
    /* Error if function is set for gate types other than LUT. */
    LOAD_STATE_RETURN_ON_ERROR(type != LUT && func != 0, doc);

    /* Parse input gates. */
    int inp = 0;
    gatenum inputs[] = {NO_GATE, NO_GATE, NO_GATE};
    for (xmlNodePtr input = gate->children; input != NULL; input = input->next) {
      if (strcmp((char*)input->name, "input") != 0) {
        continue;
      }
      char *gatestr = (char*)xmlGetProp(input, (xmlChar*)"gate");
      char *endptr;
      int gatenum = strtoul(gatestr, &endptr, 10);
      if (*endptr != '\0') {
        xmlFree(gatestr);
        LOAD_STATE_RETURN_ON_ERROR(true, doc);
      }
      xmlFree(gatestr);
      gatestr = NULL;
      LOAD_STATE_RETURN_ON_ERROR(gatenum >= st.num_gates, doc);
      inputs[inp++] = gatenum;
    }

    ttable table;
    if (type <= TRUE_GATE) {
      LOAD_STATE_RETURN_ON_ERROR(inp != 2, doc);
      table = generate_ttable_2(type, st.gates[inputs[0]].table, st.gates[inputs[1]].table);
    } else if (type == NOT) {
      LOAD_STATE_RETURN_ON_ERROR(inp != 1, doc);
      table = ~st.gates[inputs[0]].table;
    } else if (type == IN) {
      LOAD_STATE_RETURN_ON_ERROR(inp != 0, doc);
      LOAD_STATE_RETURN_ON_ERROR(st.num_gates >= 8, doc);
      LOAD_STATE_RETURN_ON_ERROR(st.num_gates != 0 && st.gates[st.num_gates - 1].type != IN, doc);
      table = generate_target(st.num_gates, false);
    } else if (type == LUT) {
      LOAD_STATE_RETURN_ON_ERROR(inp != 3, doc);
      table = generate_lut_ttable(func, st.gates[inputs[0]].table, st.gates[inputs[1]].table,
          st.gates[inputs[2]].table);
    } else {
      LOAD_STATE_RETURN_ON_ERROR(true, doc);
    }

    st.gates[st.num_gates].table = table;
    st.gates[st.num_gates].type = type;
    st.gates[st.num_gates].in1 = inputs[0];
    st.gates[st.num_gates].in2 = inputs[1];
    st.gates[st.num_gates].in3 = inputs[2];
    st.gates[st.num_gates].function = (uint8_t)func;
    st.num_gates += 1;
  }

  /* Parse outputs. */
  for (xmlNodePtr output = gates->children; output != NULL; output = output->next) {
    if (strcmp((char*)output->name, "output") != 0) {
      continue;
    }
    char *bitstr = (char*)xmlGetProp(output, (xmlChar*)"bit");
    char *endptr;
    int bit = strtoul(bitstr, &endptr, 10);
    if (*endptr != '\0') {
      xmlFree(bitstr);
      LOAD_STATE_RETURN_ON_ERROR(true, doc);
    }
    xmlFree(bitstr);
    bitstr = NULL;
    LOAD_STATE_RETURN_ON_ERROR(bit >= 8, doc);
    LOAD_STATE_RETURN_ON_ERROR(st.outputs[bit] != NO_GATE, doc);

    char *gatestr = (char*)xmlGetProp(output, (xmlChar*)"gate");
    int gate = strtoul(gatestr, &endptr, 10);
    if (*endptr != '\0') {
      xmlFree(gatestr);
      LOAD_STATE_RETURN_ON_ERROR(true, doc);
    }
    xmlFree(gatestr);
    gatestr = NULL;
    LOAD_STATE_RETURN_ON_ERROR(gate >= st.num_gates, doc);

    st.outputs[bit] = gate;
  }

  xmlFreeDoc(doc);

  /* Calculate SAT metric. */
  for (int i = 0; i < st.num_gates; i++) {
    if (st.gates[i].type == LUT) {
      st.sat_metric = 0;
      break;
    }
    st.sat_metric += get_sat_metric(st.gates[i].type);
  }

  *return_state = st;

  return true;
}
