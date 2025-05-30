/***************************************************************************
* Copyright ESIEE Paris (2020)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#pragma once

#include "pybind11/pybind11.h"
namespace py_tree_monotonic_regression {
    void py_init_tree_monotonic_regression(pybind11::module &m);
}
