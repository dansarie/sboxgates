/* state.h

   Function definitions for state.h.

   Copyright (c) 2016-2017, 2020-2021 Marcus Dansarie

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

#ifndef __STATE_H__
#define __STATE_H__

#include <stdbool.h>
#include <stdint.h>

#define MAX_GATES 500

/* Returned by functions returning a gate number to indicate that no gate was found or no gate
   could be added. */
#define NO_GATE ((gatenum)-1)

/* Used in printf format strings. */
#define PRIgatenum PRIu16

/* All two-input boolean gates and the special gates IN and LUT. */
typedef enum {
  FALSE_GATE,
  AND,
  A_AND_NOT_B,
  A,
  NOT_A_AND_B,
  B,
  XOR,
  OR,
  NOR,
  XNOR,
  NOT_B,
  A_OR_NOT_B,
  NOT_A,
  NOT_A_OR_B,
  NAND,
  TRUE_GATE,
  NOT,
  IN,
  LUT,
  END = 0xff
} gate_type;

typedef enum {GATES, SAT} metric;

/* Display strings for the gate types in gate_type. */
extern const char* const gate_name[];

/* 256 bit truth table. */
#define TABLE_SIZE 256
typedef uint64_t ttable
    __attribute((aligned(TABLE_SIZE / 8)))
    __attribute((vector_size(TABLE_SIZE / 8)));

typedef uint16_t gatenum;

typedef struct {
  ttable table;     /* The truth table of the gate. */
  gate_type type;   /* The type of gate represented. */
  gatenum in1;      /* Input 1 to the gate. NO_GATE for the inputs. */
  gatenum in2;      /* Input 2 to the gate. NO_GATE for NOT gates and the inputs. */
  gatenum in3;      /* Input 3 if LUT or NO_GATE. */
  uint8_t function; /* For LUTs: the implemented lookup table/function. */
} gate;

typedef struct {
  int max_sat_metric;    /* Current maximum accepted SAT metric. */
  int sat_metric;        /* SAT metric of the current state. */
  gatenum max_gates;     /* Current maximum accepted number of gates. */
  gatenum num_gates;     /* Current number of gates. */
  gatenum outputs[8];    /* Gate number of the respective output gates, or NO_GATE. */
  gate gates[MAX_GATES]; /* Individual gates in the current graph. */
} state;

/* Saves the state st to a file named O-GGG-MMMM-NNNNNNNN-FFFFFFFF.xml, where
   O        is the number of output Boolean functions in the circuit;
   GGG      is the number of gates in the circuit;
   MMMM     is the value of the SAT metric for the circuit;
   NNNNNNNN are the bit numbers of the output Boolean functions, in order of inclusion; and
   FFFFFFFF is a fingerprint that aims to uniquely identify the solution.
   */
void save_state(state st);

/* Returns the SAT metric of the specified gate type. Calling this with the LUT
   gate type will cause an assertion to fail. */
int get_sat_metric(gate_type type);

/* Returns the number of input gates in the state.
   st - pointer to a state. */
int get_num_inputs(const state *st);

/* Calculates the truth table of a LUT given its function and three input truth tables. */
ttable generate_lut_ttable(const uint8_t function, const ttable in1, const ttable in2,
    const ttable in3);

/* Generates a target truth table for the search.
   bit  - which bit of the input/sbox to generate the target truth table for.
   sbox - If true, a target truth table for the given bit of g_sbox_enc is generated.
          If false, the truth table of the given input bit is generated. */
ttable generate_target(uint8_t bit, bool sbox);

/* Loads a saved state from an XML file. Returns true if successful and false otherwise.
   name  - the file name to load the file from.
   state - a pointer to an allocted state struct that should be updated with the loaded state. */
bool load_state(const char *name, state *return_state);

#endif /* __STATE_H__ */
