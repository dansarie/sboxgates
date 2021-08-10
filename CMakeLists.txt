# CMakeLists.txt
#
# Copyright (c) 2019-2021 Marcus Dansarie
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

cmake_minimum_required (VERSION 3.9)

project(SBOXGATES VERSION 1.0.0 LANGUAGES C)

option(ENABLE_COVERAGE "Compile and link with gcov." OFF)

find_package(MPI REQUIRED)
find_package(LibXml2 REQUIRED)

add_executable(sboxgates boolfunc.c convert_graph.c lut.c sboxgates.c state.c)
include_directories(${LIBXML2_INCLUDE_DIR})
target_include_directories(sboxgates PRIVATE ${MPI_C_INCLUDE_PATH})
target_link_libraries(sboxgates ${MPI_C_LIBRARIES} ${MPI_C_LINK_FLAGS} ${LIBXML2_LIBRARIES})

set(CMAKE_C_FLAGS "-march=native -Ofast -g -Wall -Wpedantic")
if (ENABLE_COVERAGE AND CMAKE_C_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
endif (ENABLE_COVERAGE AND CMAKE_C_COMPILER_ID STREQUAL "GNU")

include(CheckIPOSupported)
check_ipo_supported(RESULT result)
if(result)
  set_target_properties(sboxgates PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

install(TARGETS sboxgates DESTINATION bin)
