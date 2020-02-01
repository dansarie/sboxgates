/* state.c

   Helper functions for saving and loading files containing logic circuit
   representations of S-boxes created by sboxgates.

   Copyright (c) 2016-2017, 2020 Marcus Dansarie

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
#include <string.h>
#include "lut.h"
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
    char *type = NULL;
    switch (st.gates[i].type) {
      case IN:
        type = "IN";
        break;
      case NOT:
        type = "NOT";
        break;
      case AND:
        type = "AND";
        break;
      case OR:
        type = "OR";
        break;
      case XOR:
        type = "XOR";
        break;
      case ANDNOT:
        type = "ANDNOT";
        break;
      case LUT:
        type = "LUT";
        break;
      default:
        assert(0);
    }
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

#define LOAD_STATE_RETURN_ON_ERROR(X)\
  if (X) {\
    fprintf(stderr, "Error when parsing XML document. (state.c:%d)\n", __LINE__);\
    if (doc != NULL) xmlFreeDoc(doc);\
    return false;\
  }

/* Loads a saved state */
bool load_state(const char *name, state *return_state) {
  assert(name != NULL);
  assert(return_state != NULL);

  xmlDocPtr doc = xmlParseFile(name);
  LOAD_STATE_RETURN_ON_ERROR(doc == NULL);

  /* Get gates. */
  xmlNodePtr gates = NULL;
  for (xmlNodePtr ptr = doc->children; ptr != NULL; ptr = ptr->next) {
    if (strcmp((char*)ptr->name, "gates") == 0) {
      gates = ptr;
      break;
    }
  }
  LOAD_STATE_RETURN_ON_ERROR(gates == NULL);

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
    LOAD_STATE_RETURN_ON_ERROR(typestr == NULL);
    gate_type type;
    if (strcmp(typestr, "IN") == 0) {
      type = IN;
    } else if (strcmp(typestr, "NOT") == 0) {
      type = NOT;
    } else if (strcmp(typestr, "AND") == 0) {
      type = AND;
    } else if (strcmp(typestr, "OR") == 0) {
      type = OR;
    } else if (strcmp(typestr, "ANDNOT") == 0) {
      type = ANDNOT;
    } else if (strcmp(typestr, "XOR") == 0) {
      type = XOR;
    } else if (strcmp(typestr, "LUT") == 0) {
      type = LUT;
    } else {
      xmlFree(typestr);
      LOAD_STATE_RETURN_ON_ERROR(TRUE);
    }
    xmlFree(typestr);
    typestr = NULL;

    /* Parse LUT function. */
    long func = 0;
    char *funcstr = (char*)xmlGetProp(gate, (xmlChar*)"function");
    if (funcstr != NULL) {
      func = strtol(funcstr, NULL, 16);
      xmlFree(funcstr);
      funcstr = NULL;
      LOAD_STATE_RETURN_ON_ERROR(func <= 0 || func > 255);
    }
    /* Error if function is set for gate types other than LUT. */
    LOAD_STATE_RETURN_ON_ERROR(type != LUT && func != 0);

    /* Parse input gates. */
    int inp = 0;
    gatenum inputs[] = {NO_GATE, NO_GATE, NO_GATE};
    for (xmlNodePtr input = gate->children; input != NULL; input = input->next) {
      if (strcmp((char*)input->name, "input") != 0) {
        continue;
      }
      char *gatestr = (char*)xmlGetProp(input, (xmlChar*)"gate");
      int gatenum = atoi(gatestr);
      xmlFree(gatestr);
      gatestr = NULL;
      LOAD_STATE_RETURN_ON_ERROR(gatenum < 0 || gatenum >= st.num_gates);
      inputs[inp++] = gatenum;
    }

    ttable table;
    if (type == IN) {
      LOAD_STATE_RETURN_ON_ERROR(inp != 0);
      LOAD_STATE_RETURN_ON_ERROR(st.num_gates >= 8);
      LOAD_STATE_RETURN_ON_ERROR(st.num_gates != 0 && st.gates[st.num_gates - 1].type != IN);
      table = generate_target(st.num_gates, false);
    } else if (type == NOT) {
      LOAD_STATE_RETURN_ON_ERROR(inp != 1);
      table = ~st.gates[inputs[0]].table;
    } else if (type == AND) {
      LOAD_STATE_RETURN_ON_ERROR(inp != 2);
      table = st.gates[inputs[0]].table & st.gates[inputs[1]].table;
    } else if (type == OR) {
      LOAD_STATE_RETURN_ON_ERROR(inp != 2);
      table = st.gates[inputs[0]].table | st.gates[inputs[1]].table;
    } else if (type == ANDNOT) {
      LOAD_STATE_RETURN_ON_ERROR(inp != 2);
      table = ~st.gates[inputs[0]].table & st.gates[inputs[1]].table;
    } else if (type == XOR) {
      LOAD_STATE_RETURN_ON_ERROR(inp != 2);
      table = st.gates[inputs[0]].table ^ st.gates[inputs[1]].table;
    } else if (type == LUT) {
      LOAD_STATE_RETURN_ON_ERROR(inp != 3);
      table = generate_lut_ttable(func, st.gates[inputs[0]].table, st.gates[inputs[1]].table,
          st.gates[inputs[2]].table);
    } else {
      LOAD_STATE_RETURN_ON_ERROR(TRUE);
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
    int bit = atoi(bitstr);
    xmlFree(bitstr);
    bitstr = NULL;
    LOAD_STATE_RETURN_ON_ERROR(bit < 0 || bit >= 8);
    LOAD_STATE_RETURN_ON_ERROR(st.outputs[bit] != NO_GATE);

    char *gatestr = (char*)xmlGetProp(output, (xmlChar*)"gate");
    int gate = atoi(gatestr);
    xmlFree(gatestr);
    gatestr = NULL;
    LOAD_STATE_RETURN_ON_ERROR(gate < 0 || gate >= st.num_gates);

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
