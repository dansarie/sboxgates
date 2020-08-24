/* boolfunc.c

   Copyright (c) 2020 Marcus Dansarie

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
#include <string.h>
#include "boolfunc.h"

uint8_t get_val(uint8_t fun, uint8_t bit) {
  assert(fun < 16);
  return (fun >> (3 - bit)) & 1;
}

static bool inarray(uint8_t fun, const boolfunc * array) {
  for (int i = 0; array[i].num_inputs != 0; i++) {
    if (array[i].fun == fun) {
      return true;
    }
  }
  return false;
}

int get_not_functions(const boolfunc * restrict input_funs, boolfunc * restrict output_funs) {
  assert(input_funs != NULL);
  assert(output_funs != NULL);

  output_funs[0].num_inputs = 0;

  int outp = 0;
  for (int i = 0; input_funs[i].num_inputs != 0; i++) {
    uint8_t cfun = ~input_funs[i].fun & 0xF;
    if (!inarray(cfun, input_funs) && !inarray(cfun, output_funs)) {
      output_funs[outp] = input_funs[i];
      output_funs[outp].fun = cfun;
      output_funs[outp].not_out = !output_funs[outp].not_out;
      outp += 1;
      output_funs[outp].num_inputs = 0;
    }
  }
  return outp;
}

boolfunc create_2_input_fun(uint8_t fun) {
  assert(fun < 16);
  boolfunc ret;
  ret.num_inputs = 2;
  ret.fun = fun;
  ret.fun1 = fun;
  ret.fun2 = NO_GATE;
  ret.not_a = false;
  ret.not_b = false;
  ret.not_c = false;
  ret.not_out = false;
  ret.ab_commutative = ~(fun >> 1 ^ fun >> 2) & 1;
  ret.ac_commutative = false;
  ret.bc_commutative = false;
  return ret;
}

int get_3_input_function_list(const boolfunc * restrict input_funs,
    boolfunc * restrict output_funs, bool try_nots) {
  assert(input_funs != NULL);
  assert(output_funs != NULL);
  boolfunc funs[256];
  memset(funs, 0xff, sizeof(boolfunc) * 256);


  uint8_t nots[] = {0, 1, 2, 4, 3, 5, 6, 7};
  /* Iterate over all combinations of two two-input boolean functions. */
  for (int notsp = 0; notsp < (try_nots ? 8 : 1); notsp++) {
    for (int i = 0; input_funs[i].num_inputs != 0; i++) {
      for (int k = 0; input_funs[k].num_inputs != 0; k++) {
        assert(input_funs[k].num_inputs == 2);
        assert(input_funs[k].fun == input_funs[k].fun1);
        assert(input_funs[k].fun < 16);
        uint8_t fun = 0;
        /* Compute truth table. */
        for (uint8_t val = 0; val < 8; val++) {
          uint8_t ab = ((7 - val) ^ nots[notsp]) >> 1;
          uint8_t c = ((7 - val) ^ nots[notsp]) & 1;
          fun <<= 1;
          fun |= get_val(input_funs[k].fun, get_val(input_funs[i].fun, ab) << 1 | c);
        }
        if (funs[fun].fun >= 16) { /* If function isn't already set. */
          funs[fun].num_inputs = 3;
          funs[fun].fun = fun;
          funs[fun].fun1 = input_funs[i].fun;
          funs[fun].fun2 = input_funs[k].fun;
          funs[fun].not_a = (nots[notsp] & 4) != 0;
          funs[fun].not_b = (nots[notsp] & 2) != 0;
          funs[fun].not_c = (nots[notsp] & 1) != 0;
          funs[fun].not_out = false;
          funs[fun].ab_commutative = ~(fun >> 2 ^ fun >> 4) & ~(fun >> 3 ^ fun >> 5) & 1;
          funs[fun].ac_commutative = ~(fun >> 1 ^ fun >> 4) & ~(fun >> 3 ^ fun >> 6) & 1;
          funs[fun].bc_commutative = ~(fun >> 1 ^ fun >> 2) & ~(fun >> 5 ^ fun >> 6) & 1;
        }
      }
    }
  }

  /* Attempt to create new functions by appending a NOT gate to the output of those already
     discovered. */
  if (try_nots) {
    for (int i = 0; i < 256; i++) {
      int nfun = ~i & 0xff;
      if (funs[i].fun1 < 16 && funs[nfun].fun1 >= 16) {
        funs[nfun] = funs[i];
        funs[nfun].fun = ~funs[nfun].fun;
        funs[nfun].not_out = true;
      }
    }
  }

  int outp = 0;
  for (int i = 0; i < 256; i++) {
    if (funs[i].fun1 < 16) {
      output_funs[outp++] = funs[i];
    }
  }
  return outp;
}

ttable generate_ttable_2(const gate_type gate, const ttable in1, const ttable in2) {
  ttable zero = {0};
  switch (gate) {
    case FALSE_GATE:  return zero;
    case AND:         return in1 & in2;
    case A_AND_NOT_B: return in1 & ~in2;
    case A:           return in1;
    case NOT_A_AND_B: return ~in1 & in2;
    case B:           return in2;
    case XOR:         return in1 ^ in2;
    case OR:          return in1 | in2;
    case NOR:         return ~(in1 | in2);
    case XNOR:        return (in1 & in2) | (~in1 & ~in2);
    case NOT_B:       return ~in2;
    case A_OR_NOT_B:  return in1 | ~in2;
    case NOT_A:       return ~in1;
    case NOT_A_OR_B:  return ~in1 | in2;
    case NAND:        return ~(in1 & in2);
    case TRUE_GATE:   return ~zero;
    default:          assert(0);
  }
}

ttable generate_ttable_3(boolfunc fun, const ttable in1, const ttable in2, const ttable in3) {
  ttable ret = {0};
  if (fun.fun & 1) {
    ret |= ~in1 & ~in2 & ~in3;
  }
  if (fun.fun & 2) {
    ret |= ~in1 & ~in2 & in3;
  }
  if (fun.fun & 4) {
    ret |= ~in1 & in2 & ~in3;
  }
  if (fun.fun & 8) {
    ret |= ~in1 & in2 & in3;
  }
  if (fun.fun & 16) {
    ret |= in1 & ~in2 & ~in3;
  }
  if (fun.fun & 32) {
    ret |= in1 & ~in2 & in3;
  }
  if (fun.fun & 64) {
    ret |= in1 & in2 & ~in3;
  }
  if (fun.fun & 128) {
    ret |= in1 & in2 & in3;
  }
  return ret;
}
