/*
 * sboxgates.h
 * Copyright (c) 2016-2017 Marcus Dansarie
 */

#include <inttypes.h>
#include <stdbool.h>
#include <x86intrin.h>

#define MSGPACK_FORMAT_VERSION 2
#define MAX_GATES 500
#define NO_GATE ((gatenum)-1)
#define PRIgatenum PRIu16

typedef enum {IN, NOT, AND, OR, XOR, ANDNOT, LUT} gate_type;
typedef enum {GATES, SAT} metric;

typedef __m256i ttable; /* 256 bit truth table. */
typedef uint16_t gatenum;

typedef struct {
  ttable table;
  gate_type type;
  gatenum in1; /* Input 1 to the gate. NO_GATE for the inputs. */
  gatenum in2; /* Input 2 to the gate. NO_GATE for NOT gates and the inputs. */
  gatenum in3; /* Input 3 if LUT or NO_GATE. */
  uint8_t function; /* For LUTs. */
} gate;

typedef struct {
  int max_sat_metric;
  int sat_metric;
  gatenum max_gates;
  gatenum num_gates;  /* Current number of gates. */
  gatenum outputs[8]; /* Gate number of the respective output gates, or NO_GATE. */
  gate gates[MAX_GATES];
} state;

uint32_t state_fingerprint(const state st);
void save_state(state st);
bool load_state(const char *name, state *return_state);

