/* convert_graph.h

   Header file for graph conversion functions.

   Copyright (c) 2019-2021 Marcus Dansarie

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

#ifndef __CONVERT_GRAPH_H__
#define __CONVERT_GRAPH_H__

#include "state.h"

/* Prints a truth table to the console. Used for debugging.
   tbl - the truth table to print. */
void print_ttable(ttable tbl);

/* Prints a gate network to stdout in Graphviz dot format.
   st - pointer to the state to be printed. */
void print_digraph(const state *st);

/* Converts a gate network to a C or CUDA function and prints it to stdout. If the state contains
   at least one LUT gate it will be converted to a CUDA function. Otherwise, it will be converted to
   a C function.
   st - pointer to the state to be converted to a function. */
bool print_c_function(const state *st);

#endif /* __CONVERT_GRAPH_H__ */
