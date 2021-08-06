/* lut.h

   Header file for LUT functions.

   Copyright (c) 2019-2020 Marcus Dansarie

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

#ifndef __LUT_H__
#define __LUT_H__

#include "sboxgates.h"
#include "state.h"

/* Returns true if it is possible to create a num input Boolean function with the specified input
   truth tables that satisfies the target truth table, under the specified mask.*/
bool check_n_lut_possible(const int num, const ttable target, const ttable mask,
    const ttable *tables);

/* Generates all possible truth tables for a LUT with the given three input truth tables. Used for
   caching in the search functions. */
void generate_lut_ttables(const ttable in1, const ttable in2, const ttable in3, ttable *out);

/* Returns a LUT function func with the three input truth tables with an output truth table matching
   target in the positions where mask is set. Returns true on success and false if no function that
   can satisfy the target truth table exists. */
bool get_lut_function(const ttable in1, const ttable in2, const ttable in3, const ttable target,
    const ttable mask, const bool randomize, uint8_t *func);

/* Search for a combination of five outputs in the graph that can be connected with a 5-input LUT
   to create an output truth table that matches target in the positions where mask is set. Returns
   true on success. In that case the result is returned in the 7 position array ret: ret[0]
   contains the outer LUT function, ret[1] the inner LUT function, and ret[2] - ret[6] the five
   input gate numbers. */
bool search_5lut(const state st, const ttable target, const ttable mask, const int8_t *inbits,
    uint16_t *ret, int verbosity);

/* Search for a combination of seven outputs in the graph that can be connected with a 7-input LUT
   to create an output truth table that matches target in the positions where mask is set. Returns
   true on success. In that case the result is returned in the 10 position array ret: ret[0]
   contains the outer LUT function, ret[1] the middle LUT function, ret[2] the inner LUT function,
   and ret[3] - ret[9] the seven input gate numbers. */
bool search_7lut(const state st, const ttable target, const ttable mask, const int8_t *inbits,
    uint16_t *ret, int verbosity);

gatenum lut_search(state *st, const ttable target, const ttable mask, const int8_t *inbits,
    const gatenum *gate_order, const options *opt);

#endif /* __LUT_H__ */
