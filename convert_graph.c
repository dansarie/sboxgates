/* convert_graph.c

   Helper functions for converting generated graphs to C/CUDA code or Graphviz dot format for
   visualization.

   Copyright (c) 2016-2017, 2019-2021 Marcus Dansarie

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
#include "sboxgates.h"

void print_ttable(ttable tbl) {
  uint64_t vec[4];
  memcpy((ttable*)vec, &tbl, sizeof(ttable));
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

void print_digraph(const state *st) {
  printf("digraph sbox {\n");
  assert(st->num_gates < MAX_GATES);
  for (int gt = 0; gt < st->num_gates; gt++) {
    char gatename[20];
    assert(st->gates[gt].type <= LUT);
    if (st->gates[gt].type == IN) {
      sprintf(gatename, "IN %d", gt);
    } else if (st->gates[gt].type == LUT) {
      sprintf(gatename, "0x%02x", st->gates[gt].function);
    } else {
      strcpy(gatename, gate_name[st->gates[gt].type]);
      for (int i = 0; gatename[i] != '\0'; i++) {
        if (gatename[i] == '_') {
          gatename[i] = ' ';
        }
      }
    }
    printf("  gt%d [label=\"%s\"];\n", gt, gatename);
  }
  for (int gt = get_num_inputs(st); gt < st->num_gates; gt++) {
    if (st->gates[gt].in1 != NO_GATE) {
      printf("  gt%" PRIgatenum " -> gt%d;\n", st->gates[gt].in1, gt);
    }
    if (st->gates[gt].in2 != NO_GATE) {
      printf("  gt%" PRIgatenum " -> gt%d;\n", st->gates[gt].in2, gt);
    }
    if (st->gates[gt].in3 != NO_GATE) {
      printf("  gt%" PRIgatenum " -> gt%d;\n", st->gates[gt].in3, gt);
    }
  }
  for (uint8_t i = 0; i < 8; i++) {
    if (st->outputs[i] != NO_GATE) {
      printf("  gt%" PRIgatenum " -> out%" PRIu8 ";\n", st->outputs[i], i);
    }
  }
  printf("}\n");
}

/* Called by print_c_function to get variable names. Returns true if the variable should be
   declared.
   st      - pointer to state.
   gate    - gate to generate variable name for.
   buf     - output buffer.
   ptr_out - true if output variables are pointers (i.e. there is more than one). */
static bool get_c_variable_name(const state * restrict st, const gatenum gate, char * restrict buf,
    bool ptr_out) {
  if (gate < get_num_inputs(st)) {
    sprintf(buf, "in.b%" PRIgatenum, gate);
    return false;
  }
  for (uint8_t i = 0; i < get_num_inputs(st); i++) {
    if (st->outputs[i] == gate) {
      sprintf(buf, "%sout%d", ptr_out ? "*" : "", i);
      return false;
    }
  }
  sprintf(buf, "var%" PRIgatenum, gate);
  return true;
}

bool print_c_function(const state *st) {
  /* Generate CUDA code if LUT gates are present. */
  bool cuda = false;
  for (int gate = get_num_inputs(st); gate < st->num_gates; gate++) {
    if (st->gates[gate].type == LUT) {
      cuda = true;
      break;
    }
  }

  int num_outputs = 0;
  int outp_num = 0;
  for (int outp = 0; outp < get_num_inputs(st); outp++) {
    if (st->outputs[outp] != NO_GATE) {
      num_outputs += 1;
      outp_num = outp;
    }
  }
  if (num_outputs <= 0) {
    fprintf(stderr, "Error: no output gates in circuit. (convert_graph.c:%d)\n", __LINE__);
    return false;
  }
  bool ptr_ret = num_outputs > 1;

  /* Generate type definitions. */
  const char TYPE[] = "bit_t";
  if (cuda) {
    printf("#define LUT(a,b,c,d,e) asm(\"lop3.b32 %%0, %%1, %%2, %%3, \"#e\";\" : "
        "\"=r\"(a): \"r\"(b), \"r\"(c), \"r\"(d));\n");
    printf("typedef int %s;\n", TYPE);
  } else {
    printf("typedef unsigned long long int %s;\n", TYPE);
  }
  printf("typedef struct {\n");
  for (int i = 0; i < get_num_inputs(st); i++) {
    printf("  %s b%d;\n", TYPE, i);
  }
  printf("} bits;\n");

  /* Output start of S-box function. */
  if (cuda) {
    if (num_outputs > 1) {
      printf("__device__ __forceinline__ void s(bits in");
      for (int outp = 0; outp < 8; outp++) {
        if (st->outputs[outp] != NO_GATE) {
          printf(", %s *out%d", TYPE, outp);
        }
      }
      printf(") {\n");
    } else {
      printf("__device__ __forceinline__ %s s%d(bits in) {\n", TYPE, outp_num);
    }
  } else {
    if (num_outputs > 1) {
      printf("void s(bits in");
      for (int outp = 0; outp < get_num_inputs(st); outp++) {
        if (st->outputs[outp] != NO_GATE) {
          printf(", %s *out%d", TYPE, outp);
        }
      }
      printf(") {\n");
    } else {
      printf("%s s%d(bits in) {\n", TYPE, outp_num);
    }
  }

  /* Output graph code. */
  char start[10];
  char var_in1[10];
  char var_in2[10];
  char var_in3[10];
  char var_out[10];
  for (int gate = get_num_inputs(st); gate < st->num_gates; gate++) {
    if (st->gates[gate].in1 != NO_GATE) {
      get_c_variable_name(st, st->gates[gate].in1, var_in1, ptr_ret);
    }
    if (st->gates[gate].in2 != NO_GATE) {
      get_c_variable_name(st, st->gates[gate].in2, var_in2, ptr_ret);
    }
    if (st->gates[gate].in3 != NO_GATE) {
      get_c_variable_name(st, st->gates[gate].in3, var_in3, ptr_ret);
    }
    bool decl = get_c_variable_name(st, gate, var_out, ptr_ret);
    if (decl || var_out[0] != '*') {
      sprintf(start, "  %s ", TYPE);
    } else {
      strcpy(start, "  ");
    }

    switch (st->gates[gate].type) {
      case FALSE_GATE:  printf("%s%s = 0;\n", start, var_out);                            break;
      case AND:         printf("%s%s = %s & %s;\n", start, var_out, var_in1, var_in2);    break;
      case A_AND_NOT_B: printf("%s%s = %s & ~%s;\n", start, var_out, var_in1, var_in2);   break;
      case A:           printf("%s%s = %s;\n", start, var_out, var_in1);                  break;
      case NOT_A_AND_B: printf("%s%s = ~%s & %s;\n", start, var_out, var_in1, var_in2);   break;
      case B:           printf("%s%s = %s;\n", start, var_out, var_in2);                  break;
      case XOR:         printf("%s%s = %s ^ %s;\n", start, var_out, var_in1, var_in2);    break;
      case OR:          printf("%s%s = %s | %s;\n", start, var_out, var_in1, var_in2);    break;
      case NOR:         printf("%s%s = ~(%s | %s);\n", start, var_out, var_in1, var_in2); break;
      case XNOR:        printf("%s%s = (%s & %s) | (~%s & ~%s);\n", start, var_out, var_in1,
                            var_in2, var_in1, var_in2);                                   break;
      case NOT_B:       printf("%s%s = ~%s;\n", start, var_out, var_in2);                 break;
      case A_OR_NOT_B:  printf("%s%s = %s | ~%s;\n", start, var_out, var_in1, var_in2);   break;
      case NOT_A:       printf("%s%s = ~%s;\n", start, var_out, var_in1);                 break;
      case NOT_A_OR_B:  printf("%s%s = ~%s | %s;\n", start, var_out, var_in1, var_in2);   break;
      case NAND:        printf("%s%s = ~(%s & %s);\n", start, var_out, var_in1, var_in2); break;
      case TRUE_GATE:   printf("%s%s = ~0;\n", start, var_out);                           break;
      case NOT:         printf("%s%s = ~%s;\n", start, var_out, var_in1);                 break;
      case LUT:         printf("  %s %s; LUT(%s, %s, %s, %s, 0x%02x);\n", TYPE, var_out, var_out,
          var_in1, var_in2, var_in3, st->gates[gate].function); break;
      default:          assert(0);
    }

    if (!decl && num_outputs == 1) {
      get_c_variable_name(st, gate, var_out, ptr_ret);
      printf("  return %s;\n", var_out);
    }
  }
  printf("}\n");
  return true;
}
