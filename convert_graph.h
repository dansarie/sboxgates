/* convert_graph.h

   Header file for graph conversion functions.

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

#ifndef __CONVERT_GRAPH_H__
#define __CONVERT_GRAPH_H__

#include "state.h"

/* Prints a truth table to the console. Used for debugging. */
void print_ttable(ttable tbl);

/* Prints a gate network to stdout in Graphviz dot format. */
void print_digraph(const state st);

/* Converts a gate network to a C function and prints it to stdout. */
bool print_c_function(const state st);

#endif /* __CONVERT_GRAPH_H__ */
