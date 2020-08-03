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

static bool inarray(uint8_t fun, const boolfunc * array, int num_inputs) {
  for (int i = 0; i < num_inputs; i++) {
    if (array[i].fun == fun) {
      return true;
    }
  }
  return false;
}

int get_not_functions(const boolfunc * restrict input_funs, int num_inputs,
    boolfunc * restrict output_funs) {
  assert(input_funs != NULL);
  assert(num_inputs >= 0);
  assert(output_funs != NULL);

  int outp = 0;
  for (int i = 0; i < num_inputs; i++) {
    uint8_t cfun = ~input_funs[i].fun;
    if (!inarray(cfun, input_funs, num_inputs)) {
      output_funs[outp] = input_funs[i];
      output_funs[outp].fun = cfun;
      output_funs[outp].not = !output_funs[outp].not;
      outp += 1;
    }
  }
  return outp;
}

boolfunc create_2_input_fun(uint8_t fun) {
  assert(fun < 16);
  boolfunc ret;
  ret.fun = fun;
  ret.fun1 = fun;
  ret.fun2 = 0xff;
  ret.not = false;
  ret.ab_commutative = ~(fun >> 1 ^ fun >> 2) & 1;
  ret.ac_commutative = false;
  ret.bc_commutative = false;
  return ret;
}

int get_3_input_function_list(const boolfunc * restrict input_funs, int num_inputs,
    boolfunc * restrict output_funs) {
  assert(input_funs != NULL);
  assert(num_inputs >= 0);
  assert(output_funs != NULL);
  boolfunc funs[256];
  memset(funs, 0xff, sizeof(boolfunc) * 256);

  /* Iterate over all combinations of two two-input boolean functions. */
  for (int i = 0; i < num_inputs; i++) {
    for (int k = 0; k < num_inputs; k++) {
      assert(input_funs[k].fun < 16);
      uint8_t fun = 0;
      /* Compute truth table. */
      for (uint8_t val = 0; val < 8; val++) {
        uint8_t ab = val >> 1;
        uint8_t c = val & 1;
        fun <<= 1;
        fun |= get_val(input_funs[k].fun, get_val(input_funs[i].fun, ab) << 1 | c);
      }
      if (funs[fun].fun1 >= 16) { /* If function isn't already set. */
        funs[fun].fun = fun;
        funs[fun].fun1 = input_funs[i].fun;
        funs[fun].fun2 = input_funs[k].fun;
        funs[fun].not = false;
        funs[fun].ab_commutative = ~(fun >> 2 ^ fun >> 4) & ~(fun >> 3 ^ fun >> 5) & 1;
        funs[fun].ac_commutative = ~(fun >> 1 ^ fun >> 4) & ~(fun >> 3 ^ fun >> 6) & 1;
        funs[fun].bc_commutative = ~(fun >> 1 ^ fun >> 2) & ~(fun >> 5 ^ fun >> 6) & 1;
      }
    }
  }

  /* Attempt to create new functions by appending a NOT gate to the output of those already
     discovered. */
  for (int i = 0; i < 256; i++) {
    int nfun = ~i & 0xff;
    if (funs[i].fun1 < 16 && funs[nfun].fun1 >= 16) {
      funs[nfun] = funs[i];
      funs[nfun].not = true;
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
