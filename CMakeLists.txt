# CMakeLists.txt
#
# Copyright (c) 2019 Marcus Dansarie
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

cmake_minimum_required (VERSION 3.10)

project(SBOXGATES VERSION 1.0.0 LANGUAGES C)

find_package(MPI REQUIRED)
find_library(MSGPACK_LIBRARY msgpackc)

add_executable(sboxgates convert_graph.c lut.c sboxgates.c state.c)
target_include_directories(sboxgates PRIVATE ${MPI_C_INCLUDE_PATH})
target_link_libraries(sboxgates ${MPI_C_LIBRARIES} ${MPI_C_LINK_FLAGS} ${MSGPACK_LIBRARY})

set(CMAKE_C_FLAGS "-march=native -Ofast -g -Wall -Wpedantic")

install(TARGETS sboxgates DESTINATION bin)
