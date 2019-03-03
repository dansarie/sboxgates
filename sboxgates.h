/* sboxgates.h

   Copyright (c) 2019 Marcus Dansarie

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

#ifndef __SBOXGATES_H__
#define __SBOXGATES_H__

#include "state.h"

/* Performs a masked test for equality. Only bits set to 1 in the mask will be tested. */
bool ttable_equals_mask(const ttable in1, const ttable in2, const ttable mask);

/* Returns the number of input gates in the state. */
int get_num_inputs(const state *st);

/* Generates pseudorandom 64 bit strings. Used for randomizing the search process. */
uint64_t xorshift1024();

#endif /* __SBOXGATES_H__ */
