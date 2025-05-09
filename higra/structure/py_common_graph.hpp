/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#pragma once

#include "../py_common.hpp"
#include "higra/graph.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

namespace py_common_graph {
    template<typename T>
    using pyarray = xt::pyarray<T>;
    using namespace hg;

    template<typename graph_t>
    struct def_out_degree {
        template<typename value_t, typename C>
        static
        void def(C &c, const char *doc) {
            c.def("out_degree", [](graph_t &g,
                                   pyarray<value_t> vertices) {
                      hg_assert_vertex_indices(g, vertices);
                      return hg::out_degree(vertices, g);
                  },
                  doc,
                  pybind11::arg("vertices_array"));
        }
    };

    template<typename T>
    auto cpp_edge_2_python(const T &e);

    template<>
    inline
    auto cpp_edge_2_python(const std::pair<hg::index_t, hg::index_t> &e) {
        return pybind11::make_tuple(e.first, e.second);
    }

    template<>
    inline
    auto cpp_edge_2_python(const hg::indexed_edge<hg::index_t, hg::index_t> &e) {
        return pybind11::make_tuple(e.first, e.second, e.index);
    }

    template<typename graph_t, typename pyc>
    void add_incidence_graph_concept(pyc &c) {
        using edge_t = typename hg::graph_traits<graph_t>::edge_descriptor;
        using vertex_t = typename hg::graph_traits<graph_t>::vertex_descriptor;
        using iterator_transform_function = std::function<pybind11::tuple(edge_t)>;
        using out_edge_iterator = hg::transform_forward_iterator<iterator_transform_function,
                typename hg::graph_traits<graph_t>::out_edge_iterator,
                pybind11::tuple
        >;

        c.def("out_edges", [](const graph_t &g,
                              const vertex_t v) {
                  hg_assert_vertex_index(g, v);
                  auto it = hg::out_edges(v, g);
                  // wrapping out edge iterator to python friendly type
                  auto it1 = out_edge_iterator(it.first, cpp_edge_2_python<edge_t>);
                  auto it2 = out_edge_iterator(it.second, cpp_edge_2_python<edge_t>);
                  return pybind11::make_iterator(it1, it2);

              },
              "Iterator over all out edges from 'vertex'. An out edge is a tuple '(vertex, adjacent_vertex)'.",
              pybind11::arg("vertex"));

        c.def("out_degree", [](graph_t &g, vertex_t vertex) {
                  hg_assert_vertex_index(g, vertex);
                  return hg::out_degree(vertex, g);
              },
              "Return the out degree of the given vertex.",
              pybind11::arg("vertex"));

        add_type_overloads<def_out_degree<graph_t>, int, unsigned int, long long, unsigned long long>
                (c, "Return the out degree of the given vertices.");


    };


    template<typename graph_t>
    struct def_degree {
        template<typename value_t, typename C>
        static
        void def(C &c, const char *doc) {
            c.def("degree", [](graph_t &g,
                               pyarray<value_t> vertices) {
                      hg_assert_vertex_indices(g, vertices);
                      return hg::degree(vertices, g);
                  },
                  doc,
                  pybind11::arg("vertices_array"));
        }
    };

    template<typename graph_t>
    struct def_in_degree {
        template<typename value_t, typename C>
        static
        void def(C &c, const char *doc) {
            c.def("in_degree", [](graph_t &g,
                                  pyarray<value_t> vertices) {
                      hg_assert_vertex_indices(g, vertices);
                      return hg::in_degree(vertices, g);
                  },
                  doc,
                  pybind11::arg("vertices_array"));
        }
    };

    template<typename graph_t, typename pyc>
    void add_bidirectionnal_graph_concept(pyc &c) {
        using edge_t = typename hg::graph_traits<graph_t>::edge_descriptor;
        using vertex_t = typename hg::graph_traits<graph_t>::vertex_descriptor;
        using iterator_transform_function = std::function<pybind11::tuple(edge_t)>;
        using in_edge_iterator = hg::transform_forward_iterator<
                iterator_transform_function,
                typename hg::graph_traits<graph_t>::in_edge_iterator,
                pybind11::tuple>;

        c.def("in_edges", [](graph_t &g,
                             const vertex_t v) {
                  hg_assert_vertex_index(g, v);
                  auto it = hg::in_edges(v, g);
                  // wrapping in edge iterator to python friendly type
                  auto it1 = in_edge_iterator(it.first, cpp_edge_2_python<edge_t>);
                  auto it2 = in_edge_iterator(it.second, cpp_edge_2_python<edge_t>);
                  return pybind11::make_iterator(it1, it2);

              },
              "Iterator over all in edges from 'vertex'. An in edge is a tuple '(adjacent_vertex, vertex)'.",
              pybind11::arg("vertex"));

        c.def("degree", [](graph_t &g, vertex_t vertex) {
                  hg_assert_vertex_index(g, vertex);
                  return hg::degree(vertex, g);
              },
              "Return the degree of the given vertex.",
              pybind11::arg("vertex"));

        add_type_overloads<def_degree<graph_t>, int, unsigned int, long long, unsigned long long>
                (c, "Return the degree of the given vertices.");

        c.def("in_degree", [](graph_t &g, vertex_t vertex) {
                  hg_assert_vertex_index(g, vertex);
                  return hg::in_degree(vertex, g);
              },
              "Return the in degree of the given vertex.",
              pybind11::arg("vertex"));

        add_type_overloads<def_in_degree<graph_t>, int, unsigned int, long long, unsigned long long>
                (c, "Return the in degree of the given vertices.");
    }

    template<typename graph_t, typename pyc>
    void add_adjacency_graph_concept(pyc &c) {
        using vertex_t = typename hg::graph_traits<graph_t>::vertex_descriptor;

        c.def("adjacent_vertices", [](graph_t &g,
                                      const vertex_t v) {
                  hg_assert_vertex_index(g, v);
                  auto it = hg::adjacent_vertices(v, g);
                  return pybind11::make_iterator(it.first, it.second);
              },
              "Iterator over all vertices adjacent to the given vertex.",
              pybind11::arg("vertex"));
    }

    template<typename graph_t, typename pyc>
    void add_vertex_list_graph_concept(pyc &c) {

        c.def("vertices", [](graph_t &g) {
                  auto it = hg::vertices(g);
                  return pybind11::make_iterator(it.first, it.second);
              },
              "Iterator over all vertices of the graph.");

        c.def("num_vertices", [](graph_t &g) { return hg::num_vertices(g); },
              "Return the number of vertices in the graph");
    }

    template<typename graph_t, typename pyc>
    void add_edge_list_graph_concept(pyc &c) {
        using edge_t = typename hg::graph_traits<graph_t>::edge_descriptor;
        using iterator_transform_function = std::function<pybind11::tuple(edge_t)>;
        using edge_iterator = hg::transform_forward_iterator<
                iterator_transform_function,
                typename hg::graph_traits<graph_t>::edge_iterator,
                pybind11::tuple>;
        c.def("edges", [](graph_t &g) {
                  auto it = hg::edges(g);
                  // wrapping  edge iterator to python friendly type
                  auto it1 = edge_iterator(it.first, cpp_edge_2_python<edge_t>);
                  auto it2 = edge_iterator(it.second, cpp_edge_2_python<edge_t>);
                  return pybind11::make_iterator(it1, it2);
              },
              "Iterator over all edges of the graph.");

        c.def("num_edges", [](graph_t &g) { return hg::num_edges(g); },
              "Return the number of edges in the graph");
    }


    template<typename graph_t, typename pyc>
    void add_edge_accessor_graph_concept(pyc &c) {

        c.def("source", [](const graph_t &g, const pybind11::tuple *v) {
                  return (*v)[0];
              },

              "Get the source vertex of an edge.",
              pybind11::arg("edge"));

        c.def("target", [](const graph_t &g, const pybind11::tuple *v) {
                  return (*v)[1];
              },

              "Get the target vertex of an edge.",
              pybind11::arg("edge"));
    }

    template<typename graph_t, typename pyc>
    void add_edge_index_graph_concept(pyc &c) {
        using vertex_t = typename hg::graph_traits<graph_t>::vertex_descriptor;
        using edge_index_t = typename hg::graph_traits<graph_t>::edge_index;

        c.def("edge_from_index", [](graph_t &g, edge_index_t v) {
                  hg_assert_edge_index(g, v);
                  auto e = hg::edge_from_index(v, g);
                  return cpp_edge_2_python(e);
              },

              "Get an edge from its index.",
              pybind11::arg("edge_index"));

        c.def("index", [](const graph_t &g, const pybind11::tuple *v) {
                  return (*v)[2];
              },

              "Get the index of an edge.",
              pybind11::arg("edge"));
    }
}