/* convert_graph.c

   Helper functions for converting generated graphs to C/CUDA code or Graphviz dot format for
   visualization.

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
#include <stdio.h>
#include <string.h>
#include "convert_graph.h"

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
static bool get_c_variable_name(const state st, const gatenum gate, char *buf, bool ptr_out) {
  if (gate < 8) {
    sprintf(buf, "in.b%" PRIgatenum, gate);
    return false;
  }
  for (uint8_t i = 0; i < 8; i++) {
    if (st.outputs[i] == gate) {
      sprintf(buf, "%sout%d", ptr_out ? "*" : "", i);
      return false;
    }
  }
  sprintf(buf, "var%" PRIgatenum, gate);
  return true;
}

/* Converts a gate network to a C function and prints it to stdout. */
bool print_c_function(const state st) {
  bool cuda = false;
  for (int gate = 8; gate < st.num_gates; gate++) {
    if (st.gates[gate].type == LUT) {
      cuda = true;
      break;
    }
  }

  int num_outputs = 0;
  int outp_num = 0;
  for (int outp = 0; outp < 8; outp++) {
    if (st.outputs[outp] != NO_GATE) {
      num_outputs += 1;
      outp_num = outp;
    }
  }
  if (num_outputs <= 0) {
    fprintf(stderr, "Error: no output gates in circuit.\n");
    return false;
  }
  bool ptr_ret = num_outputs > 1;

  #define TYPE_STR "bit_t"
  const char TYPE[] = TYPE_STR;
  const char STRUCT_STR[] = "typedef struct {\n"
                            "  " TYPE_STR " b0;\n"
                            "  " TYPE_STR " b1;\n"
                            "  " TYPE_STR " b2;\n"
                            "  " TYPE_STR " b3;\n"
                            "  " TYPE_STR " b4;\n"
                            "  " TYPE_STR " b5;\n"
                            "  " TYPE_STR " b6;\n"
                            "  " TYPE_STR " b7;\n"
                            "} eightbits;\n";
  if (cuda) {
    printf("#define LUT(a,b,c,d,e) asm(\"lop3.b32 %%0, %%1, %%2, %%3, \"#e\";\" : \"=r\"(##a): "
        "\"r\"(##b), \"r\"(##c), \"r\"(##d));\n");
    printf("typedef uint %s;\n", TYPE);
    printf(STRUCT_STR);
    if (num_outputs > 1) {
      printf("__device__ __forceinline__ void s(eightbits in");
      for (int outp = 0; outp < 8; outp++) {
        if (st.outputs[outp] != NO_GATE) {
          printf(", %s *out%d", TYPE, outp);
        }
      }
      printf(") {\n");
    } else {
      printf("__device__ __forceinline__ %s s%d(eightbits in) {\n", TYPE, outp_num);
    }
  } else {
    printf("typedef __m256i %s;\n", TYPE);
    printf(STRUCT_STR);
    if (num_outputs > 1) {
      printf("static inline void s(eightbits in");
      for (int outp = 0; outp < 8; outp++) {
        if (st.outputs[outp] != NO_GATE) {
          printf(", %s *out%d", TYPE, outp);
        }
      }
      printf(") {\n");
    } else {
      printf("static inline %s s%d(eightbits in) {\n", TYPE, outp_num);
    }
  }
  char buf[10];
  for (int gate = 8; gate < st.num_gates; gate++) {
    bool ret = get_c_variable_name(st, gate, buf, ptr_ret);
    if (ret != true) {
      printf("  %s = ", buf);
    } else {
      printf("  %s %s = ", TYPE, buf);
    }
    if (st.gates[gate].type == LUT) {
      printf("LUT(%s, ", buf);
      get_c_variable_name(st, st.gates[gate].in1, buf, ptr_ret);
      printf("%s, ", buf);
      get_c_variable_name(st, st.gates[gate].in2, buf, ptr_ret);
      printf("%s, ", buf);
      get_c_variable_name(st, st.gates[gate].in3, buf, ptr_ret);
      printf("%s, 0x%02x);\n", buf, st.gates[gate].function);
    } else {
      get_c_variable_name(st, st.gates[gate].in1, buf, ptr_ret);
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
      get_c_variable_name(st, st.gates[gate].in2, buf, ptr_ret);
      printf("%s;\n", buf);
    }
    if (!ret && num_outputs == 1) {
      get_c_variable_name(st, gate, buf, ptr_ret);
      printf("  return %s;\n", buf);
    }
  }
  printf("}\n");
  return true;
}
