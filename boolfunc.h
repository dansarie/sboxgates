/* boolfunc.h

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

#ifndef __BOOLFUNC_H__
#define __BOOLFUNC_H__

#include <inttypes.h>
#include <stdbool.h>
#include "state.h"

/* Three-input boolean functions f(A, B, C) are created from two-input boolean functions as:
   fun(A, B, C) = fun2(fun1(A, B), C). */

typedef struct {
  int num_inputs;
  uint8_t fun;         /* Three-input boolean function. fun2(fun1(0xF0, 0xCC), 0xAA) */
  gate_type fun1;      /* Two-input boolean function 1. */
  gate_type fun2;      /* Two-input boolean function 2. */
  bool not_a;          /* True if NOT gate is appended to input A. */
  bool not_b;          /* True if NOT gate is appended to input B. */
  bool not_c;          /* True if NOT gate is appended to input C. */
  bool not_out;        /* True if NOT gate is appended to output. */
  bool ab_commutative; /* True if the function is commutative with respect to inputs A and B. */
  bool ac_commutative; /* True if the function is commutative with respect to inputs A and C. */
  bool bc_commutative; /* True if the function is commutative with respect to inputs B and C. */
} boolfunc;

/* Returns the value of the two-input boolean function fun for inputs bit = A << 1 | B. */
uint8_t get_val(uint8_t fun, uint8_t bit);

/* Returns a boolfunc struct representing the two-input boolean function fun. */
boolfunc create_2_input_fun(uint8_t fun);

/* Generates a list of new functions by appending a NOT gate to one or the inputs or the output of
   the functions in input_funs.
   input_funs  - array of input functions, terminated with END.
   output_funs - output_array. Will contain num_inputs members at most on return. */
int get_not_functions(const boolfunc * restrict input_funs,
    boolfunc * restrict output_funs);

/* Generates a list of unique three-input boolean functions from a list of available two-input
   boolean functions. Returns the number of functions in output_fun.
   input_funs  - array of input functions, terminated with END.
   output_funs - output array. Will contain num_inputs^2 members at most on return.
   try_nots    - set to true to append NOT gates in order to find more 3-input functions. */
int get_3_input_function_list(const boolfunc * restrict input_funs,
    boolfunc * restrict output_funs, bool try_nots);

/* Generates the truth table for a two-input gate.
   fun - the gate function.
   in1 - input truth table 1.
   in2 - input truth table 2. */
ttable generate_ttable_2(const gate_type gate, const ttable in1, const ttable in2);

/* Generates the truth table for a three-input gate.
   fun - the gate function.
   in1 - input truth table 1.
   in2 - input truth table 2.
   in3 - input truth table 3. */
ttable generate_ttable_3(boolfunc fun, const ttable in1, const ttable in2, const ttable in3);

#endif /* __BOOLFUNC_H__ */
