/* lut.c

   Functions for handling and search for LUTs.

   Copyright (c) 2016-2017, 2019-2020 Marcus Dansarie

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
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lut.h"

static void get_nth_combination(int64_t n, int num_gates, int t, gatenum first, gatenum *ret);
static bool get_search_result(uint16_t *ret, int *quit_msg, MPI_Request *recv_req,
    MPI_Request *send_req);
static inline int64_t n_choose_k(int n, int k);
static inline void next_combination(gatenum *combination, int t, int max);

/* Called by check_n_lut_possible. */
static bool check_n_lut_possible_recurse(const int num, const ttable target, const ttable mask,
    const ttable *tables, ttable *match, ttable tt) {

  if (num == 0) {
    if (ttable_equals_mask(target & tt, tt, mask)) {
      *match |= tt;
    } else if (!ttable_zero(target & tt & mask)) {
      return false;
    }
    return true;
  }

  if (!check_n_lut_possible_recurse(num - 1, target, mask, tables + 1, match, tt & ~tables[0])) {
    return false;
  }
  if (!check_n_lut_possible_recurse(num - 1, target, mask, tables + 1, match, tt & tables[0])) {
    return false;
  }

  return true;
}

/* Returns true if it is possible to create a num input Boolean function with the specified input
   truth tables that satisfies the target truth table, under the specified mask.*/
bool check_n_lut_possible(const int num, const ttable target, const ttable mask,
    const ttable *tables) {
  ttable match = {0};
  ttable tt = ~match;
  if (!check_n_lut_possible_recurse(num, target, mask, tables, &match, tt)) {
    return false;
  }
  return ttable_equals_mask(target, match, mask);
}

/* Generates all possible truth tables for a LUT with the given three input truth tables. Used for
   caching in the search functions. */
void generate_lut_ttables(const ttable in1, const ttable in2, const ttable in3, ttable *out) {
  for (int func = 0; func < 256; func++) {
    out[func] = generate_lut_ttable(func, in1, in2, in3);
  }
}

/* Returns a LUT function func with the three input truth tables with an output truth table matching
   target in the positions where mask is set. Returns true on success and false if no function that
   can satisfy the target truth table exists. */
bool get_lut_function(ttable in1, ttable in2, ttable in3, ttable target, ttable mask,
    const bool randomize, uint8_t *func) {
  *func = 0;
  uint64_t funcset = 0; /* Keeps track of which function bits have been set. */

  while (!ttable_zero(mask)) {
    for (int v = 0; v < sizeof(ttable) / sizeof(uint64_t); v++) {
      if (mask[v] & 1) {
        uint64_t temp = ((in1[v] & 1) << 2) | ((in2[v] & 1) << 1) | (in3[v] & 1);
        if ((funcset & (1 << temp)) == 0) {
          *func |= (target[v] & 1) << temp;
          funcset |= 1 << temp;
        } else if ((*func & (1 << temp)) != ((target[v] & 1) << temp)) {
          return false;
        }
      }
    }
    target >>= 1;
    mask >>= 1;
    in1 >>= 1;
    in2 >>= 1;
    in3 >>= 1;
  }

  /* Randomize don't-cares in table. */
  if (randomize && funcset != 0xff) {
    *func |= ~funcset & (uint8_t)xorshift1024();
  }

  return true;
}

/* Search for a combination of five outputs in the graph that can be connected with a 5-input LUT
   to create an output truth table that matches target in the positions where mask is set. Returns
   true on success. In that case the result is returned in the 7 position array ret: ret[0]
   contains the outer LUT function, ret[1] the inner LUT function, and ret[2] - ret[6] the five
   input gate numbers. */
bool search_5lut(const state st, const ttable target, const ttable mask, const int8_t *inbits,
    uint16_t *ret, int verbosity) {
  assert(ret != NULL);
  assert(st.num_gates >= 5);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  uint8_t func_order[256];
  for (int i = 0; i < 256; i++) {
    func_order[i] = i;
  }
  /* Fisher-Yates shuffle. */
  for (int i = 0; i < 256; i++) {
    uint64_t j = xorshift1024() % (i + 1);
    uint8_t t = func_order[i];
    func_order[i] = func_order[j];
    func_order[j] = t;
  }

  /* Determine this rank's work. */
  uint64_t search_space_size = n_choose_k(st.num_gates, 5);
  uint64_t worker_space_size = search_space_size / size;
  uint64_t remainder = search_space_size - worker_space_size * size;
  uint64_t start_n;
  uint64_t stop_n;
  if (rank < remainder) {
    start_n = (worker_space_size + 1) * rank;
    stop_n = start_n + worker_space_size + 1;
  } else {
    start_n = (worker_space_size + 1) * remainder + worker_space_size * (rank - remainder);
    stop_n = start_n + worker_space_size;
  }

  MPI_Request recv_req = MPI_REQUEST_NULL;
  MPI_Request send_req = MPI_REQUEST_NULL;
  int quit_msg = -1;

  if (rank == 0) {
    MPI_Irecv(&quit_msg, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &recv_req);
  } else {
    MPI_Irecv(&quit_msg, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &recv_req);
  }

  if (start_n >= n_choose_k(st.num_gates, 5)) {
    return get_search_result(ret, &quit_msg, &recv_req, &send_req);
  }

  gatenum nums[5] = {NO_GATE, NO_GATE, NO_GATE, NO_GATE, NO_GATE};
  get_nth_combination(start_n, st.num_gates, 5, 0, nums);

  ttable tt[5] = {st.gates[nums[0]].table, st.gates[nums[1]].table, st.gates[nums[2]].table,
      st.gates[nums[3]].table, st.gates[nums[4]].table};

  memset(ret, 0, sizeof(uint16_t) * 10);

  bool quit = false;
  for (uint64_t i = start_n; !quit && i < stop_n; i++) {
    /* Reject input gate combinations that contain a bit that the algorithm has already used as a
       multiplexer input in step 5 of the algorithm. */
    bool rejected = false;
    for (int k = 0; !rejected && inbits[k] != -1; k++) {
      for (int m = 0; m < 5; m++) {
        if (nums[m] == inbits[k]) {
          rejected = true;
          break;
        }
      }
    }

    if (!rejected && check_n_lut_possible(5, target, mask, tt)) {
      /* Try all 10 ways to build a 5LUT from two 3LUTs. */
      gatenum order[5] = {0, 1, 2, 3, 4};
      for (int k = 0; k < 10; k++) {
        for (uint16_t fo = 0; !quit && fo < 256; fo++) {
          uint8_t func_outer = func_order[fo];
          ttable t_outer = generate_lut_ttable(func_outer, tt[order[0]], tt[order[1]],
              tt[order[2]]);
          uint8_t func_inner;
          if (!get_lut_function(t_outer, tt[order[3]], tt[order[4]], target, mask, true,
              &func_inner)) {
            continue;
          }
          ttable t_inner = generate_lut_ttable(func_inner, t_outer, tt[order[3]], tt[order[4]]);
          assert(ttable_equals_mask(target, t_inner, mask));
          ret[0] = func_outer;
          ret[1] = func_inner;
          ret[2] = nums[order[0]];
          ret[3] = nums[order[1]];
          ret[4] = nums[order[2]];
          ret[5] = nums[order[3]];
          ret[6] = nums[order[4]];
          ret[7] = 0;
          ret[8] = 0;
          ret[9] = 0;
          assert(send_req == MPI_REQUEST_NULL);
          if (rank == 0) {
            quit_msg = 0;
          } else {
            MPI_Isend(&rank, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &send_req);
          }
          quit = true;
          if (verbosity >= 1) {
            printf("[% 4d] Found 5LUT: %02x %02x    %3d %3d %3d %3d %3d\n", rank, ret[0],
                ret[1], ret[2], ret[3], ret[4], ret[5], ret[6]);
          }
        }
        next_combination(order, 3, 5); /* Next combination of three gates. */
        /* Work out the other two gates. */
        unsigned int xx = ~((1 << order[0]) | (1 << order[1]) | (1 << order[2]));
        order[3] = __builtin_ffs(xx) - 1;
        xx ^= 1 << order[3];
        order[4] = __builtin_ffs(xx) - 1;
      }
    }

    if (!quit) {
      int flag;
      MPI_Test(&recv_req, &flag, MPI_STATUS_IGNORE);
      if (flag) {
        break;
      }
      next_combination(nums, 5, st.num_gates);
      tt[0] = st.gates[nums[0]].table;
      tt[1] = st.gates[nums[1]].table;
      tt[2] = st.gates[nums[2]].table;
      tt[3] = st.gates[nums[3]].table;
      tt[4] = st.gates[nums[4]].table;
    }
  }

  return get_search_result(ret, &quit_msg, &recv_req, &send_req);
}

/* Search for a combination of seven outputs in the graph that can be connected with a 7-input LUT
   to create an output truth table that matches target in the positions where mask is set. Returns
   true on success. In that case the result is returned in the 10 position array ret: ret[0]
   contains the outer LUT function, ret[1] the middle LUT function, ret[2] the inner LUT function,
   and ret[3] - ret[9] the seven input gate numbers. */
bool search_7lut(const state st, const ttable target, const ttable mask, const int8_t *inbits,
    uint16_t *ret, int verbosity) {
  assert(ret != NULL);
  assert(st.num_gates >= 7);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* Determine this rank's work. */
  uint64_t search_space_size = n_choose_k(st.num_gates, 7);
  uint64_t worker_space_size = search_space_size / size;
  uint64_t remainder = search_space_size - worker_space_size * size;
  uint64_t start;
  uint64_t stop;
  if (rank < remainder) {
    start = (worker_space_size + 1) * rank;
    stop = start + worker_space_size + 1;
  } else {
    start = (worker_space_size + 1) * remainder + worker_space_size * (rank - remainder);
    stop = start + worker_space_size;
  }

  gatenum nums[7];
  if (start >= n_choose_k(st.num_gates, 7)) {
    memset(nums, 0, sizeof(gatenum) * 7);
  } else {
    get_nth_combination(start, st.num_gates, 7, 0, nums);
  }

  ttable tt[7] = {st.gates[nums[0]].table, st.gates[nums[1]].table, st.gates[nums[2]].table,
      st.gates[nums[3]].table, st.gates[nums[4]].table, st.gates[nums[5]].table,
      st.gates[nums[6]].table};

  /* Filter out the gate combinations where a 7LUT is possible. */
  gatenum *result = malloc(sizeof(gatenum) * 7 * 100000);
  assert(result != NULL);
  int p = 0;
  for (uint64_t i = start; i < stop; i++) {
    /* Reject input gate combinations that contain a bit that the algorithm has already used as a
       multiplexer input in step 5 of the algorithm. */
    bool rejected = false;
    for (int k = 0; !rejected && inbits[k] != -1; k++) {
      for (int m = 0; m < 7; m++) {
        if (nums[m] == inbits[k]) {
          rejected = true;
          break;
        }
      }
    }

    if (!rejected && check_n_lut_possible(7, target, mask, tt)) {
      result[p++] = nums[0];
      result[p++] = nums[1];
      result[p++] = nums[2];
      result[p++] = nums[3];
      result[p++] = nums[4];
      result[p++] = nums[5];
      result[p++] = nums[6];
    }
    if (p >= 7 * 100000) {
      break;
    }
    next_combination(nums, 7, st.num_gates);
    tt[0] = st.gates[nums[0]].table;
    tt[1] = st.gates[nums[1]].table;
    tt[2] = st.gates[nums[2]].table;
    tt[3] = st.gates[nums[3]].table;
    tt[4] = st.gates[nums[4]].table;
    tt[5] = st.gates[nums[5]].table;
    tt[6] = st.gates[nums[6]].table;
  }

  /* Gather the number of hits for each rank.*/
  int rank_nums[size];
  MPI_Allgather(&p, 1, MPI_INT, rank_nums, 1, MPI_INT, MPI_COMM_WORLD);
  assert(rank_nums[0] % 7 == 0);
  int tsize = rank_nums[0];
  int offsets[size];
  offsets[0] = 0;
  for (int i = 1; i < size; i++) {
    assert(rank_nums[i] % 7 == 0);
    tsize += rank_nums[i];
    offsets[i] = offsets[i - 1] + rank_nums[i - 1];
  }

  gatenum *lut_list = malloc(sizeof(gatenum) * tsize);
  assert(lut_list != NULL);

  /* Get all hits. */
  MPI_Allgatherv(result, p, MPI_UINT16_T, lut_list, rank_nums, offsets, MPI_UINT16_T,
      MPI_COMM_WORLD);
  free(result);
  result = NULL;

  /* Calculate rank's work chunk. */
  worker_space_size = (tsize / 7) / size;
  remainder = (tsize / 7) - worker_space_size * size;
  if (rank < remainder) {
    start = (worker_space_size + 1) * rank;
    stop  = start + worker_space_size + 1;
  } else {
    start = (worker_space_size + 1) * remainder + worker_space_size * (rank - remainder);
    stop = start + worker_space_size;
  }

  uint8_t outer_func_order[256];
  uint8_t middle_func_order[256];
  for (int i = 0; i < 256; i++) {
    outer_func_order[i] = middle_func_order[i] = i;
  }

  /* Fisher-Yates shuffle the function search orders. */
  for (int i = 0; i < 256; i++) {
    uint64_t oj = xorshift1024() % (i + 1);
    uint64_t mj = xorshift1024() % (i + 1);
    uint8_t ot = outer_func_order[i];
    uint8_t mt = middle_func_order[i];
    outer_func_order[i] = outer_func_order[oj];
    middle_func_order[i] = middle_func_order[mj];
    outer_func_order[oj] = ot;
    middle_func_order[mj] = mt;
  }
  int outer_cache_set = 0;
  int middle_cache_set = 0;
  ttable outer_cache[256];
  ttable middle_cache[256];
  memset(ret, 0, 10 * sizeof(uint16_t));

  MPI_Request recv_req = MPI_REQUEST_NULL;
  MPI_Request send_req = MPI_REQUEST_NULL;
  int quit_msg = -1;

  if (rank == 0) {
    MPI_Irecv(&quit_msg, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &recv_req);
  } else {
    MPI_Irecv(&quit_msg, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &recv_req);
  }

  bool quit = false;
  const int order[70 * 7] = {
      0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 6, 5, 0, 1, 2, 3, 5, 6, 4, 0, 1, 2, 4, 5, 6, 3,
      0, 1, 3, 2, 4, 5, 6, 0, 1, 3, 2, 4, 6, 5, 0, 1, 3, 2, 5, 6, 4, 0, 1, 3, 4, 5, 6, 2,
      0, 1, 4, 2, 3, 5, 6, 0, 1, 4, 2, 3, 6, 5, 0, 1, 4, 2, 5, 6, 3, 0, 1, 4, 3, 5, 6, 2,
      0, 1, 5, 2, 3, 4, 6, 0, 1, 5, 2, 3, 6, 4, 0, 1, 5, 2, 4, 6, 3, 0, 1, 5, 3, 4, 6, 2,
      0, 1, 6, 2, 3, 4, 5, 0, 1, 6, 2, 3, 5, 4, 0, 1, 6, 2, 4, 5, 3, 0, 1, 6, 3, 4, 5, 2,
      0, 2, 3, 1, 4, 5, 6, 0, 2, 3, 1, 4, 6, 5, 0, 2, 3, 1, 5, 6, 4, 0, 2, 3, 4, 5, 6, 1,
      0, 2, 4, 1, 3, 5, 6, 0, 2, 4, 1, 3, 6, 5, 0, 2, 4, 1, 5, 6, 3, 0, 2, 4, 3, 5, 6, 1,
      0, 2, 5, 1, 3, 4, 6, 0, 2, 5, 1, 3, 6, 4, 0, 2, 5, 1, 4, 6, 3, 0, 2, 5, 3, 4, 6, 1,
      0, 2, 6, 1, 3, 4, 5, 0, 2, 6, 1, 3, 5, 4, 0, 2, 6, 1, 4, 5, 3, 0, 2, 6, 3, 4, 5, 1,
      0, 3, 4, 1, 2, 5, 6, 0, 3, 4, 1, 2, 6, 5, 0, 3, 4, 1, 5, 6, 2, 0, 3, 4, 2, 5, 6, 1,
      0, 3, 5, 1, 2, 4, 6, 0, 3, 5, 1, 2, 6, 4, 0, 3, 5, 1, 4, 6, 2, 0, 3, 5, 2, 4, 6, 1,
      0, 3, 6, 1, 2, 4, 5, 0, 3, 6, 1, 2, 5, 4, 0, 3, 6, 1, 4, 5, 2, 0, 3, 6, 2, 4, 5, 1,
      0, 4, 5, 1, 2, 3, 6, 0, 4, 5, 1, 2, 6, 3, 0, 4, 5, 1, 3, 6, 2, 0, 4, 5, 2, 3, 6, 1,
      0, 4, 6, 1, 2, 3, 5, 0, 4, 6, 1, 2, 5, 3, 0, 4, 6, 1, 3, 5, 2, 0, 4, 6, 2, 3, 5, 1,
      0, 5, 6, 1, 2, 3, 4, 0, 5, 6, 1, 2, 4, 3, 0, 5, 6, 1, 3, 4, 2, 0, 5, 6, 2, 3, 4, 1,
      1, 2, 3, 4, 5, 6, 0, 1, 2, 4, 3, 5, 6, 0, 1, 2, 5, 3, 4, 6, 0, 1, 2, 6, 3, 4, 5, 0,
      1, 3, 4, 2, 5, 6, 0, 1, 3, 5, 2, 4, 6, 0, 1, 3, 6, 2, 4, 5, 0, 1, 4, 5, 2, 3, 6, 0,
      1, 4, 6, 2, 3, 5, 0, 1, 5, 6, 2, 3, 4, 0
    };
  for (int i = start; !quit && i < stop; i++) {
    for (int k = 0; !quit && k < 70; k++) {
      const gatenum a = lut_list[7 * i + order[7 * k + 0]];
      const gatenum b = lut_list[7 * i + order[7 * k + 1]];
      const gatenum c = lut_list[7 * i + order[7 * k + 2]];
      const gatenum d = lut_list[7 * i + order[7 * k + 3]];
      const gatenum e = lut_list[7 * i + order[7 * k + 4]];
      const gatenum f = lut_list[7 * i + order[7 * k + 5]];
      const gatenum g = lut_list[7 * i + order[7 * k + 6]];
      const ttable ta = st.gates[a].table;
      const ttable tb = st.gates[b].table;
      const ttable tc = st.gates[c].table;
      const ttable td = st.gates[d].table;
      const ttable te = st.gates[e].table;
      const ttable tf = st.gates[f].table;
      const ttable tg = st.gates[g].table;
      if (((uint64_t)a << 32 | (uint64_t)b << 16 | c) != outer_cache_set) {
        generate_lut_ttables(ta, tb, tc, outer_cache);
        outer_cache_set = (uint64_t)a << 32 | (uint64_t)b << 16 | c;
      }
      if (((uint64_t)d << 32 | (uint64_t)e << 16 | f) != middle_cache_set) {
        generate_lut_ttables(td, te, tf, middle_cache);
        middle_cache_set = (uint64_t)d << 32 | (uint64_t)e << 16 | f;
      }

      for (uint16_t fo = 0; !quit && fo < 256; fo++) {
        uint8_t func_outer = outer_func_order[fo];
        ttable t_outer = outer_cache[func_outer];
        for (uint16_t fm = 0; !quit && fm < 256; fm++) {
          uint8_t func_middle = middle_func_order[fm];
          ttable t_middle = middle_cache[func_middle];
          uint8_t func_inner;
          if (!get_lut_function(t_outer, t_middle, tg, target, mask, true, &func_inner)) {
            continue;
          }
          ttable t_inner = generate_lut_ttable(func_inner, t_outer, t_middle, tg);
          assert(ttable_equals_mask(target, t_inner, mask));
          ret[0] = func_outer;
          ret[1] = func_middle;
          ret[2] = func_inner;
          ret[3] = a;
          ret[4] = b;
          ret[5] = c;
          ret[6] = d;
          ret[7] = e;
          ret[8] = f;
          ret[9] = g;
          assert(send_req == MPI_REQUEST_NULL);
          if (rank == 0) {
            quit_msg = 0;
          } else {
            MPI_Isend(&rank, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &send_req);
          }
          quit = true;
          if (verbosity >= 1) {
            printf("[% 4d] Found 7LUT: %02x %02x %02x %3d %3d %3d %3d %3d %3d %3d\n", rank,
                func_outer, func_middle, func_inner, a, b, c, d, e, f, g);
          }
        }
      }
      if (!quit) {
        int flag;
        MPI_Test(&recv_req, &flag, MPI_STATUS_IGNORE);
        if (flag) {
          quit = true;
        }
      }
    }
  }
  free(lut_list);
  return get_search_result(ret, &quit_msg, &recv_req, &send_req);
}

gatenum lut_search(state *st, const ttable target, const ttable mask, const int8_t *inbits,
    const gatenum *gate_order, const options *opt) {
  assert(st != NULL);
  assert(inbits != NULL);
  assert(gate_order != NULL);
  assert(opt != NULL);
  assert(opt->lut_graph);

  /* Look through all combinations of three gates in the circuit. For each combination, check if any
     of the 256 possible three bit Boolean functions produces the desired map. If so, add that LUT
     and return the ID. */

  for (int i = 0; i < st->num_gates; i++) {
    const gatenum gi = gate_order[i];
    const ttable ta = st->gates[gi].table;
    for (int k = i + 1; k < st->num_gates; k++) {
      const gatenum gk = gate_order[k];
      const ttable tb = st->gates[gk].table;
      for (int m = k + 1; m < st->num_gates; m++) {
        const gatenum gm = gate_order[m];
        const ttable tc = st->gates[gm].table;
        const ttable tables[] = {ta, tb, tc};
        if (!check_n_lut_possible(3, target, mask, tables)) {
          continue;
        }
        uint8_t func;
        if (!get_lut_function(ta, tb, tc, target, mask, opt->randomize, &func)) {
          continue;
        }
        ttable nt = generate_lut_ttable(func, ta, tb, tc);
        assert(ttable_equals_mask(target, nt, mask));
        ASSERT_AND_RETURN(add_lut(st, func, nt, gi, gk, gm), target, st, mask);
      }
    }
  }

  if (!check_num_gates_possible(st, 2, 0, opt)) {
    return NO_GATE;
  }

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* Broadcast work to be done. */
  mpi_work work;
  work.st = *st;
  work.target = target;
  work.mask = mask;
  work.quit = false;
  work.verbosity = opt->verbosity;
  memcpy(work.inbits, inbits, sizeof(uint8_t) * 8);
  MPI_Bcast(&work, 1, g_mpi_work_type, 0, MPI_COMM_WORLD);

  /* Look through all combinations of five gates in the circuit. For each combination, check if a
     combination of two of the possible 256 three bit Boolean functions as in LUT(LUT(a,b,c),d,e)
     produces the desired map. If so, add those LUTs and return the ID of the output LUT. */

  uint16_t res[10];

  memset(res, 0, sizeof(uint16_t) * 10);
  if (opt->verbosity >= 2) {
    printf("[   0] Search 5.\n");
  }

  if (work.st.num_gates >= 5
      && search_5lut(work.st, work.target, work.mask, work.inbits, res, opt->verbosity)) {
    uint8_t func_outer = (uint8_t)res[0];
    uint8_t func_inner = (uint8_t)res[1];
    gatenum a = res[2];
    gatenum b = res[3];
    gatenum c = res[4];
    gatenum d = res[5];
    gatenum e = res[6];
    ttable ta = st->gates[a].table;
    ttable tb = st->gates[b].table;
    ttable tc = st->gates[c].table;
    ttable td = st->gates[d].table;
    ttable te = st->gates[e].table;
    if (opt->verbosity >= 1) {
      printf("[   0]   Selected: %02x %02x    %3d %3d %3d %3d %3d\n",
          func_outer, func_inner, a, b, c, d, e);
    }

    const ttable tables[] = {ta, tb, tc, td, te};
    assert(check_n_lut_possible(5, target, mask, tables));
    ttable t_outer = generate_lut_ttable(func_outer, ta, tb, tc);
    ttable t_inner = generate_lut_ttable(func_inner, t_outer, td, te);
    assert(ttable_equals_mask(target, t_inner, mask));

    ASSERT_AND_RETURN(add_lut(st, func_inner, t_inner,
        add_lut(st, func_outer, t_outer, a, b, c), d, e), target, st, mask);
  }

  if (!check_num_gates_possible(st, 3, 0, opt)) {
    bool search7 = false;
    MPI_Bcast(&search7, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    return NO_GATE;
  }
  bool search7 = true;
  MPI_Bcast(&search7, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

  if (opt->verbosity >= 2) {
    printf("[   0] Search 7.\n");
  }
  if (work.st.num_gates >= 7
      && search_7lut(work.st, work.target, work.mask, work.inbits, res, opt->verbosity)) {
    uint8_t func_outer = (uint8_t)res[0];
    uint8_t func_middle = (uint8_t)res[1];
    uint8_t func_inner = (uint8_t)res[2];
    gatenum a = res[3];
    gatenum b = res[4];
    gatenum c = res[5];
    gatenum d = res[6];
    gatenum e = res[7];
    gatenum f = res[8];
    gatenum g = res[9];
    ttable ta = st->gates[a].table;
    ttable tb = st->gates[b].table;
    ttable tc = st->gates[c].table;
    ttable td = st->gates[d].table;
    ttable te = st->gates[e].table;
    ttable tf = st->gates[f].table;
    ttable tg = st->gates[g].table;
    if (opt->verbosity >= 1) {
      printf("[   0]   Selected: %02x %02x %02x %3d %3d %3d %3d %3d %3d %3d\n",
          func_outer, func_middle, func_inner, a, b, c, d, e, f, g);
    }
    const ttable tables[] = {ta, tb, tc, td, te, tf, tg};
    assert(check_n_lut_possible(7, target, mask, tables));
    ttable t_outer = generate_lut_ttable(func_outer, ta, tb, tc);
    ttable t_middle = generate_lut_ttable(func_middle, td, te, tf);
    ttable t_inner = generate_lut_ttable(func_inner, t_outer, t_middle, tg);
    assert(ttable_equals_mask(target, t_inner, mask));
    ASSERT_AND_RETURN(add_lut(st, func_inner, t_inner,
        add_lut(st, func_outer, t_outer, a, b, c),
        add_lut(st, func_middle, t_middle, d, e, f), g), target, st, mask);
  }

  if (opt->verbosity >= 2) {
    printf("[   0] No LUTs found. Num gates: %d\n", st->num_gates - get_num_inputs(st));
  }
  return NO_GATE;
}

/* Generates the nth combination of num_gates choose t gates numbered first, first + 1, ...
   Return combination in ret. */
static void get_nth_combination(int64_t n, int num_gates, int t, gatenum first, gatenum *ret) {
  assert(ret != NULL);
  assert(t <= num_gates);
  assert(n < n_choose_k(num_gates, t));

  if (t == 0) {
    return;
  }

  ret[0] = first;

  for (int i = 0; i < num_gates; i++) {
    if (n == 0) {
      for (int k = 1; k < t; k++) {
        ret[k] = ret[0] + k;
      }
      return;
    }
    int64_t nck = n_choose_k(num_gates - i - 1, t - 1);
    if (n < nck) {
      get_nth_combination(n, num_gates - ret[0] + first - 1, t - 1, ret[0] + 1, ret + 1);
      return;
    }
    ret[0] += 1;
    n -= nck;
  }
  assert(0);
}

/* Called by search_5lut and search_7lut to fetch the result of a search from the workers. */
static bool get_search_result(uint16_t *ret, int *quit_msg, MPI_Request *recv_req,
    MPI_Request *send_req) {

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int flag;
  MPI_Request *quit_requests = NULL;
  if (rank == 0) {
    /* If we've received a message, the search was successful. In that case, tell all workers to
       quit the search. */
    if (*quit_msg >= 0) {
      quit_requests = malloc(sizeof(MPI_Request) * (size - 1));
      assert(quit_requests != NULL);
      for (int i = 1; i < size; i++) {
        MPI_Isend(quit_msg, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &quit_requests[i - 1]);
      }
    }
  }

  /* Wait for all workers before continuing. */
  MPI_Barrier(MPI_COMM_WORLD);

  /* Cancel any non-completed requests. */
  if (*recv_req != MPI_REQUEST_NULL) {
    MPI_Test(recv_req, &flag, MPI_STATUS_IGNORE);
    if (!flag) {
      MPI_Cancel(recv_req);
      MPI_Wait(recv_req, MPI_STATUS_IGNORE);
    }
  }

  if (*send_req != MPI_REQUEST_NULL) {
    MPI_Test(send_req, &flag, MPI_STATUS_IGNORE);
    if (!flag) {
      MPI_Cancel(send_req);
      MPI_Wait(send_req, MPI_STATUS_IGNORE);
    }
  }

  if (quit_requests != NULL) {
    for (int i = 0; i < (size - 1); i++) {
      MPI_Test(&quit_requests[i], &flag, MPI_STATUS_IGNORE);
      if (!flag) {
        MPI_Cancel(&quit_requests[i]);
      }
    }
    MPI_Waitall(size - 1, quit_requests, MPI_STATUSES_IGNORE);
    free(quit_requests);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  /* If more than one worker found a match, there may be extra messages waiting. Receive and
     dispose of those. */
  if (rank == 0) {
    do {
      MPI_Iprobe(MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
      if (flag) {
        int foo;
        MPI_Recv(&foo, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    } while (flag);
  }

  /* Broadcast rank of worker that will broadcast search result. This will be -1 if the search
       was unsuccessful. */
  MPI_Bcast(quit_msg, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (*quit_msg < 0) {
    assert(*send_req == MPI_REQUEST_NULL);
    return false;
  }
  MPI_Bcast(ret, 10, MPI_UINT16_T, *quit_msg, MPI_COMM_WORLD);
  return true;
}

/* Creates the next combination of t numbers from the set 0, 1, ..., max - 1. */
static inline void next_combination(gatenum *combination, int t, int max) {
  int i = t - 1;
  while (i >= 0) {
    if (combination[i] + t - i < max) {
      break;
    }
    i--;
  }
  if (i < 0) {
    return;
  }
  combination[i] += 1;
  for (int k = i + 1; k < t; k++) {
    combination[k] = combination[k - 1] + 1;
  }
}

/* Calculates the binomial coefficient (n, k). */
static inline int64_t n_choose_k(int n, int k) {
  assert(n > 0);
  assert(k >= 0);
  int64_t ret = 1;
  for (int i = 1; i <= k; i++) {
    ret *= (n - i + 1);
    ret /= i;
  }
  return ret;
}
