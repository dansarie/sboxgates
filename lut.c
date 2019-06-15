/* lut.c

   Functions for handling and search for LUTs.

   Copyright (c) 2016-2017, 2019 Marcus Dansarie

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
#include "sboxgates.h"

static void get_nth_combination(int64_t n, int num_gates, int t, gatenum first, gatenum *ret);
static bool get_search_result(uint16_t *ret, int *quit_msg, MPI_Request *recv_req,
    MPI_Request *send_req);
static inline int64_t n_choose_k(int n, int k);
static inline void next_combination(gatenum *combination, int t, int max);

/* Returns true if it is possible to generate a LUT with the three input truth tables and an output
   truth table matching target in the positions where mask is set. */
bool check_3lut_possible(const ttable target, const ttable mask, const ttable t1, const ttable t2,
    const ttable t3) {
  ttable match = {0};
  ttable tt1 = ~t1;
  for (uint8_t i = 0; i < 2; i++) {
    ttable tt2 = ~t2;
    for (uint8_t k = 0; k < 2; k++) {
      ttable tt3 = ~t3;
      for (uint8_t m = 0; m < 2; m++) {
        ttable r = tt1 & tt2 & tt3;
        if (ttable_equals_mask(target & r, r, mask)) {
          match |= r;
        } else if (!ttable_zero(target & r & mask)) {
          return false;
        }
        tt3 = ~tt3;
      }
      tt2 = ~tt2;
    }
    tt1 = ~tt1;
  }
  return ttable_equals_mask(target, match, mask);
}

/* Returns true if it is possible to generate a LUT with the five input truth tables and an output
   truth table matching target in the positions where mask is set. */
bool check_5lut_possible(const ttable target, const ttable mask, const ttable t1, const ttable t2,
    const ttable t3, const ttable t4, const ttable t5) {
  ttable match = {0};
  ttable tt1 = ~t1;
  for (uint8_t i = 0; i < 2; i++) {
    ttable tt2 = ~t2;
    for (uint8_t k = 0; k < 2; k++) {
      ttable tt3 = ~t3;
      for (uint8_t m = 0; m < 2; m++) {
        ttable tt4 = ~t4;
        for (uint8_t o = 0; o < 2; o++) {
          ttable tt5 = ~t5;
          for (uint8_t q = 0; q < 2; q++) {
            ttable r = tt1 & tt2 & tt3 & tt4 & tt5;
            if (ttable_equals_mask(target & r, r, mask)) {
              match |= r;
            } else if (!ttable_zero(target & r & mask)) {
              return false;
            }
            tt5 = ~tt5;
          }
          tt4 = ~tt4;
        }
        tt3 = ~tt3;
      }
      tt2 = ~tt2;
    }
    tt1 = ~tt1;
  }
  return ttable_equals_mask(target, match, mask);
}

/* Returns true if it is possible to generate a LUT with the seven input truth tables and an output
   truth table matching target in the positions where mask is set. */
bool check_7lut_possible(const ttable target, const ttable mask, const ttable t1, const ttable t2,
    const ttable t3, const ttable t4, const ttable t5, const ttable t6, const ttable t7) {
  ttable match = {0};
  ttable tt1 = ~t1;
  for (uint8_t i = 0; i < 2; i++) {
    ttable tt2 = ~t2;
    for (uint8_t k = 0; k < 2; k++) {
      ttable tt3 = ~t3;
      for (uint8_t m = 0; m < 2; m++) {
        ttable tt4 = ~t4;
        for (uint8_t o = 0; o < 2; o++) {
          ttable tt5 = ~t5;
          for (uint8_t q = 0; q < 2; q++) {
            ttable tt6 = ~t6;
            for (uint8_t s = 0; s < 2; s++) {
              ttable tt7 = ~t7;
              for (uint8_t u = 0; u < 2; u++) {
                ttable x = tt1 & tt2 & tt3 & tt4 & tt5 & tt6 & tt7;
                if (ttable_equals_mask(target & x, x, mask)) {
                  match |= x;
                } else if (!ttable_zero(target & x & mask)) {
                  return false;
                }
                tt7 = ~tt7;
              }
              tt6 = ~tt6;
            }
            tt5 = ~tt5;
          }
          tt4 = ~tt4;
        }
        tt3 = ~tt3;
      }
      tt2 = ~tt2;
    }
    tt1 = ~tt1;
  }
  return ttable_equals_mask(target, match, mask);
}

/* Calculates the truth table of a LUT given its function and three input truth tables. */
ttable generate_lut_ttable(const uint8_t function, const ttable in1, const ttable in2,
    const ttable in3) {
  ttable ret = {0};
  if (function & 1) {
    ret |= ~in1 & ~in2 & ~in3;
  }
  if (function & 2) {
    ret |= ~in1 & ~in2 & in3;
  }
  if (function & 4) {
    ret |= ~in1 & in2 & ~in3;
  }
  if (function & 8) {
    ret |= ~in1 & in2 & in3;
  }
  if (function & 16) {
    ret |= in1 & ~in2 & ~in3;
  }
  if (function & 32) {
    ret |= in1 & ~in2 & in3;
  }
  if (function & 64) {
    ret |= in1 & in2 & ~in3;
  }
  if (function & 128) {
    ret |= in1 & in2 & in3;
  }
  return ret;
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
bool get_lut_function(const ttable in1, const ttable in2, const ttable in3, const ttable target,
    const ttable mask, const bool randomize, uint8_t *func) {
  *func = 0;
  uint8_t tableset = 0;

  uint64_t in1_v[4];
  uint64_t in2_v[4];
  uint64_t in3_v[4];
  uint64_t target_v[4];
  uint64_t mask_v[4];

  memcpy((ttable*)in1_v, &in1, sizeof(ttable));
  memcpy((ttable*)in2_v, &in2, sizeof(ttable));
  memcpy((ttable*)in3_v, &in3, sizeof(ttable));
  memcpy((ttable*)target_v, &target, sizeof(ttable));
  memcpy((ttable*)mask_v, &mask, sizeof(ttable));

  for (int v = 0; v < 4; v++) {
    for (int i = 0; i < 64; i++) {
      if (mask_v[v] & 1) {
        uint8_t temp = ((in1_v[v] & 1) << 2) | ((in2_v[v] & 1) << 1) | (in3_v[v] & 1);
        if ((tableset & (1 << temp)) == 0) {
          *func |= (target_v[v] & 1) << temp;
          tableset |= 1 << temp;
        } else if ((*func & (1 << temp)) != ((target_v[v] & 1) << temp)) {
          return false;
        }
      }
      target_v[v] >>= 1;
      mask_v[v] >>= 1;
      in1_v[v] >>= 1;
      in2_v[v] >>= 1;
      in3_v[v] >>= 1;
    }
  }

  /* Randomize don't-cares in table. */
  if (randomize && tableset != 0xff) {
    *func |= ~tableset & (uint8_t)xorshift1024();
  }

  return true;
}

/* Search for a combination of five outputs in the graph that can be connected with a 5-input LUT
   to create an output truth table that matches target in the positions where mask is set. Returns
   true on success. In that case the result is returned in the 7 position array ret: ret[0]
   contains the outer LUT function, ret[1] the inner LUT function, and ret[2] - ret[6] the five
   input gate numbers. */
bool search_5lut(const state st, const ttable target, const ttable mask, uint16_t *ret) {
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
  gatenum nums[5] = {NO_GATE, NO_GATE, NO_GATE, NO_GATE, NO_GATE};
  get_nth_combination(start_n, st.num_gates, 5, 0, nums);

  ttable tt[5] = {st.gates[nums[0]].table, st.gates[nums[1]].table, st.gates[nums[2]].table,
      st.gates[nums[3]].table, st.gates[nums[4]].table};
  gatenum cache_set[3] = {NO_GATE, NO_GATE, NO_GATE};
  ttable cache[256];

  memset(ret, 0, sizeof(uint16_t) * 10);

  MPI_Request recv_req = MPI_REQUEST_NULL;
  MPI_Request send_req = MPI_REQUEST_NULL;
  int quit_msg = -1;

  if (rank == 0) {
    MPI_Irecv(&quit_msg, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &recv_req);
  } else {
    MPI_Irecv(&quit_msg, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &recv_req);
  }

  bool quit = false;
  for (uint64_t i = start_n; !quit && i < stop_n; i++) {
    if (check_5lut_possible(target, mask, tt[0], tt[1], tt[2], tt[3], tt[4])) {
      if (cache_set[0] != nums[0] || cache_set[1] != nums[1] || cache_set[2] != nums[2]) {
        generate_lut_ttables(tt[0], tt[1], tt[2], cache);
        cache_set[0] = nums[0];
        cache_set[1] = nums[1];
        cache_set[2] = nums[2];
      }

      for (uint16_t fo = 0; !quit && fo < 256; fo++) {
        uint8_t func_outer = func_order[fo];
        ttable t_outer = cache[func_outer];
        uint8_t func_inner;
        if (!get_lut_function(t_outer, tt[3], tt[4], target, mask, true, &func_inner)) {
          continue;
        }
        ttable t_inner = generate_lut_ttable(func_inner, t_outer, tt[3], tt[4]);
        assert(ttable_equals_mask(target, t_inner, mask));
        ret[0] = func_outer;
        ret[1] = func_inner;
        ret[2] = nums[0];
        ret[3] = nums[1];
        ret[4] = nums[2];
        ret[5] = nums[3];
        ret[6] = nums[4];
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
        printf("[% 4d] Found 5LUT: %02x %02x    %3d %3d %3d %3d %3d\n", rank, func_outer,
            func_inner, nums[0], nums[1], nums[2], nums[3], nums[4]);
      }
    }
    if (!quit) {
      int flag;
      MPI_Test(&recv_req, &flag, MPI_STATUS_IGNORE);
      if (flag) {
        break;
      }
      next_combination(nums, 5, st.num_gates);
    }
  }

  return get_search_result(ret, &quit_msg, &recv_req, &send_req);
}

/* Search for a combination of seven outputs in the graph that can be connected with a 7-input LUT
   to create an output truth table that matches target in the positions where mask is set. Returns
   true on success. In that case the result is returned in the 10 position array ret: ret[0]
   contains the outer LUT function, ret[1] the middle LUT function, ret[2] the inner LUT function,
   and ret[3] - ret[9] the seven input gate numbers. */
bool search_7lut(const state st, const ttable target, const ttable mask, uint16_t *ret) {
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
  get_nth_combination(start, st.num_gates, 7, 0, nums);

  ttable tt[7] = {st.gates[nums[0]].table, st.gates[nums[1]].table, st.gates[nums[2]].table,
      st.gates[nums[3]].table, st.gates[nums[4]].table, st.gates[nums[5]].table,
      st.gates[nums[6]].table};

  /* Filter out the gate combinations where a 7LUT is possible. */
  gatenum *result = malloc(sizeof(gatenum) * 7 * 100000);
  assert(result != NULL);
  int p = 0;
  for (uint64_t i = start; i < stop; i++) {
    if (check_7lut_possible(target, mask, tt[0], tt[1], tt[2], tt[3], tt[4], tt[5], tt[6])) {
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
  for (int i = start; !quit && i < stop; i++) {
    const gatenum a = lut_list[7 * i];
    const gatenum b = lut_list[7 * i + 1];
    const gatenum c = lut_list[7 * i + 2];
    const gatenum d = lut_list[7 * i + 3];
    const gatenum e = lut_list[7 * i + 4];
    const gatenum f = lut_list[7 * i + 5];
    const gatenum g = lut_list[7 * i + 6];
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
        printf("[% 4d] Found 7LUT: %02x %02x %02x %3d %3d %3d %3d %3d %3d %3d\n", rank, func_outer,
            func_middle, func_inner, a, b, c, d, e, f, g);
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
  free(lut_list);
  return get_search_result(ret, &quit_msg, &recv_req, &send_req);
}

/* Generates the nth combination of num_gates choose t gates numbered first, first + 1, ...
   Return combination in ret. */
static void get_nth_combination(int64_t n, int num_gates, int t, gatenum first, gatenum *ret) {
  assert(ret != NULL);
  assert(t <= num_gates);

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
  MPI_Bcast(ret, 10, MPI_SHORT, *quit_msg, MPI_COMM_WORLD);
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
