/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "higra/image/graph_image.hpp"
#include "higra/hierarchy/hierarchy_core.hpp"
#include "higra/algo/tree.hpp"
#include "../test_utils.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xrandom.hpp"

namespace hierarchy_core {

    using namespace hg;
    using namespace std;

    struct _data {

        hg::tree t;

        _data() : t(xt::xarray<long>{5, 5, 6, 6, 6, 7, 7, 7}) {
        }

    } data;

    TEST_CASE("canonical binary partition tree trivial", "[hierarchy_core]") {

        auto graph = get_4_adjacency_graph({1, 2});

        array_1d<double> edge_weights{2};

        auto res = bpt_canonical(graph, edge_weights);
        auto &tree = res.tree;
        auto &altitudes = res.altitudes;
        auto &mst_edge_map = res.mst_edge_map;

        REQUIRE(num_vertices(tree) == 3);
        REQUIRE(num_edges(tree) == 2);
        REQUIRE(tree.parents() == array_1d<int>({2, 2, 2}));
        REQUIRE((altitudes == array_1d<double>({0, 0, 2})));
        REQUIRE((mst_edge_map == array_1d<int>({0})));

    }

    TEST_CASE("canonical binary partition tree", "[hierarchy_core]") {
        auto graph = get_4_adjacency_graph({2, 3});

        array_1d<double> edge_weights{1, 0, 2, 1, 1, 1, 2};

        auto res = bpt_canonical(graph, edge_weights);
        auto &tree = res.tree;
        auto &altitudes = res.altitudes;
        auto &mst_edge_map = res.mst_edge_map;

        REQUIRE(num_vertices(tree) == 11);
        REQUIRE(num_edges(tree) == 10);
        REQUIRE(xt::allclose(hg::parents(tree), xt::xarray<unsigned int>({6, 7, 9, 6, 8, 9, 7, 8, 10, 10, 10})));
        REQUIRE(xt::allclose(altitudes, xt::xarray<double>({0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2})));
        REQUIRE((mst_edge_map == array_1d<int>({1, 0, 3, 4, 2})));
    }

    TEST_CASE("boruvka binary partition tree trivial", "[hierarchy_core]")
    {
        auto graph = get_4_adjacency_graph({1, 2});

        array_1d<double> edge_weights{2};

        auto res = bpt_boruvka(graph, edge_weights);
        auto parents = res.first;
        auto mst_edge_map = res.second;

        REQUIRE(parents.size() == num_vertices(graph)+num_edges(graph));
        REQUIRE(xt::allclose(parents, xt::xarray<unsigned int>({2,2,2})));
        REQUIRE((mst_edge_map == array_1d<index_t>({in_mst})));
    }

    TEST_CASE("boruvka binary partition tree", "[hierarchy_core]")
    {
        auto graph = get_4_adjacency_graph({2, 3});

        array_1d<double> edge_weights{1, 0, 2, 1, 1, 1, 2};

        auto res = bpt_boruvka(graph, edge_weights);
        auto parents = res.first;
        auto mst_edge_map = res.second;

        auto res1 = bpt_canonical(graph, edge_weights);
        auto& tree = res1.tree;
        auto& mst_edge_map1 = res1.mst_edge_map;

        for (size_t x = 0; x < num_vertices(graph); x++)
        {
            REQUIRE(mst_edge_map1[hg::parents(tree)[x]-num_vertices(graph)] == parents[x]-num_vertices(graph));
        }
        for (long unsigned int u = num_vertices(graph); u < num_vertices(graph) * 2 - 1; u++)
        {
            auto par_k = mst_edge_map1[hg::parents(tree)[u] - num_vertices(graph)] + num_vertices(graph);
            auto par_b = parents[mst_edge_map1[u - num_vertices(graph)] + num_vertices(graph)];

            REQUIRE(par_b == par_k);
        }

        REQUIRE(parents.size() == num_vertices(graph)+num_edges(graph));
        REQUIRE(xt::allclose(parents, xt::xarray<unsigned int>({7,6,10,7,9,10,9,6,8,8,8,11,12})));
        // REQUIRE(xt::allclose(altitudes, xt::xarray<double>({0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2})));
        REQUIRE((mst_edge_map == array_1d<index_t>({in_mst,in_mst,in_mst,in_mst,in_mst,out_mst,undef_mst})));
    }

    TEST_CASE("boruvka binary partition tree paper exemple", "[hierarchy_core]")
    {
        auto graph = get_4_adjacency_graph({3, 4});

        array_1d<double> edge_weights{6, 10, 18, 1, 4, 13, 14, 16, 12, 15, 9, 2, 7, 8, 3, 17, 11};

        auto res = bpt_boruvka(graph, edge_weights);
        auto parents = res.first;
        auto mst_edge_map = res.second;

        auto res1 = bpt_canonical(graph, edge_weights);
        auto& tree = res1.tree;
        auto& mst_edge_map1 = res1.mst_edge_map;

        for (long unsigned int x = 0; x < num_vertices(graph); x++)
        {
            REQUIRE(mst_edge_map1[hg::parents(tree)[x]-num_vertices(graph)] == parents[x]-num_vertices(graph));
        }
        for (long unsigned int u = num_vertices(graph); u < num_vertices(graph) * 2 - 1; u++)
        {
            auto par_k = mst_edge_map1[hg::parents(tree)[u] - num_vertices(graph)] + num_vertices(graph);
            auto par_b = parents[mst_edge_map1[u - num_vertices(graph)] + num_vertices(graph)];

            REQUIRE(par_b == par_k);
        }

        REQUIRE(parents.size() == num_vertices(graph)+num_edges(graph));
        REQUIRE(
            xt::allclose(parents, xt::xarray<unsigned int>({12,15,16,16,13,15,23,23,26,26,24,25,22,21,14,12,17,21,18,19,
                20,21,13,24,25,17,22,27,28})));
        // REQUIRE(xt::allclose(altitudes, xt::xarray<double>({0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2})));
        REQUIRE(
            (mst_edge_map == array_1d<index_t>({in_mst,in_mst,undef_mst,in_mst,in_mst,in_mst,out_mst,out_mst,out_mst,
                in_mst,in_mst,in_mst,in_mst,in_mst,in_mst,undef_mst,out_mst})));
    }

    TEST_CASE("boruvka binary partition random graphs", "[hierarchy_core]")
    {
        auto graph = get_4_adjacency_graph({10000, 1000});

        auto mean_b = 0;
        auto mean_k = 0;
        auto nb_bench = 10;

        array_1d<double> edge_weights = xt::arange<double>(num_edges(graph));
        //random mix edge_weights
        for (auto bench = 0; bench < nb_bench; bench++)
        {
            auto seed= 1500;
            std::cout << "Seed : " << seed << std::endl;
            xt::random::seed(seed);

            //auto edge_weights = xt::concatenate(xtuple(edge_weights1,edge_weights2),0);

            xt::random::shuffle(edge_weights);

            //std::cout << edge_weights << std::endl;

            auto time_s = std::chrono::high_resolution_clock::now();
            auto res = bpt_boruvka(graph, edge_weights);
            auto time_e = std::chrono::high_resolution_clock::now();
            auto diff_time = std::chrono::duration_cast<std::chrono::milliseconds>(time_e - time_s);
            std::cout << "Time for Boruvka : " << diff_time.count() << std::endl;
            mean_b += diff_time.count();

            auto parents = res.first;
            auto mst_edge_map = res.second;

            time_s = std::chrono::high_resolution_clock::now();
            auto res1 = bpt_canonical(graph, edge_weights);
            time_e = std::chrono::high_resolution_clock::now();
            diff_time = std::chrono::duration_cast<std::chrono::milliseconds>(time_e - time_s);
            std::cout << "Time for Kruskal : " << diff_time.count() << std::endl;
            mean_k += diff_time.count();

            auto& tree = res1.tree;
            auto& mst_edge_map1 = res1.mst_edge_map;

            for (long unsigned int x = 0; x < num_vertices(graph); x++)
            {
                //std::cout << "K" << char(x+'a') << " : " << mst_edge_map1[hg::parents(tree)[x]-num_vertices(graph)] << std::endl;
                //std::cout << "B" << char(x+'a') << " : " << parents[x]-num_vertices(graph) << std::endl;
                REQUIRE(mst_edge_map1[hg::parents(tree)[x]-num_vertices(graph)] == parents[x]-num_vertices(graph));
            }

            for (long unsigned int u = num_vertices(graph); u < num_vertices(graph) * 2 - 1; u++)
            {
                //std::cout << mst_edge_map1[u-num_vertices(graph)] << " : " << mst_edge_map1[hg::parents(tree)[u]-num_vertices(graph)] << std::endl;
                auto par_k = mst_edge_map1[hg::parents(tree)[u] - num_vertices(graph)];
                auto par_b = parents[mst_edge_map1[u - num_vertices(graph)] + num_vertices(graph)] -
                    num_vertices(graph);
                REQUIRE(par_b == par_k);
            }


        }
        std::cout << "Mean time for Boruvka : " << mean_b/nb_bench << std::endl;
        std::cout << "Mean time for Kruskal : " << mean_k/nb_bench << std::endl;

    }

    TEST_CASE("simplify tree", "[hierarchy_core]") {

        auto t = data.t;

        array_1d<double> altitudes{0, 0, 0, 0, 0, 1, 2, 2};

        auto criterion = xt::equal(altitudes, xt::index_view(altitudes, t.parents()));

        auto res = hg::simplify_tree(t, criterion);
        auto nt = res.tree;
        auto nm = res.node_map;

        REQUIRE(num_vertices(nt) == 7);

        array_1d<index_t> refp{5, 5, 6, 6, 6, 6, 6};
        REQUIRE((refp == hg::parents(nt)));

        array_1d<index_t> refnm{0, 1, 2, 3, 4, 5, 7};
        REQUIRE((refnm == nm));
    }

    TEST_CASE("simplify tree remove leaves", "[hierarchy_core]") {

        tree t(xt::xarray<index_t>{8, 8, 9, 7, 7, 11, 11, 9, 10, 10, 12, 12, 12});

        array_1d<bool> criterion{false, true, true, false, false, false, false, false, true, true, false, false, false};

        auto res = hg::simplify_tree(t, criterion, true);
        auto nt = res.tree;
        auto nm = res.node_map;

        array_1d<index_t> refp{6, 5, 5, 7, 7, 6, 8, 8, 8};
        tree ref_tree(refp);
        REQUIRE(test_tree_isomorphism(nt, ref_tree));

        REQUIRE(xt::amax(xt::index_view(criterion, nm))() == false);
    }

    TEST_CASE("simplify tree remove leaves2", "[hierarchy_core]") {

        tree t(xt::xarray<index_t>{7, 7, 8, 8, 8, 9, 9, 11, 10, 10, 11, 11});

        array_1d<bool> criterion{false, false, false, true, true, true, true, false, true, false, true, false};

        auto res = hg::simplify_tree(t, criterion, true);
        auto nt = res.tree;
        auto nm = res.node_map;

        array_1d<index_t> refp{4, 4, 5, 5, 5, 5};
        tree ref_tree(refp);
        REQUIRE(test_tree_isomorphism(nt, ref_tree));

        REQUIRE(xt::amax(xt::index_view(criterion, nm))() == false);
    }

    TEST_CASE("simplify tree remove leaves3", "[hierarchy_core]") {

        tree t(xt::xarray<index_t>{7, 7, 8, 8, 8, 9, 9, 11, 10, 10, 11, 11});

        array_1d<bool> criterion{true, true, true, true, true, true, true, true, false, false, false, false};

        auto res = hg::simplify_tree(t, criterion, true);
        auto nt = res.tree;
        auto nm = res.node_map;

        array_1d<index_t> refp{2, 2, 3, 3};
        tree ref_tree(refp);
        REQUIRE(test_tree_isomorphism(nt, ref_tree));


        REQUIRE(xt::amax(xt::index_view(criterion, nm))() == false);
    }

    TEST_CASE("simplify tree remove leaves4", "[hierarchy_core]") {

        tree t(xt::xarray<index_t>{7, 7, 8, 8, 8, 9, 9, 11, 10, 10, 11, 11});

        array_1d<bool> criterion{true, true, true, true, true, true, true, true, true, false, false, false};

        auto res = hg::simplify_tree(t, criterion, true);
        auto nt = res.tree;
        auto nm = res.node_map;

        array_1d<index_t> refp{1, 2, 2};
        tree ref_tree(refp);
        REQUIRE(test_tree_isomorphism(nt, ref_tree));

        criterion(root(t)) = false;
        REQUIRE(xt::amax(xt::index_view(criterion, nm))() == false);
    }

    TEST_CASE("simplify tree remove leaves trivial", "[hierarchy_core]") {

        tree t(xt::xarray<index_t>{2, 2, 2});

        array_1d<bool> criterion{true, true, true};

        auto res = hg::simplify_tree(t, criterion, true);
        auto nt = res.tree;
        auto nm = res.node_map;

        array_1d<index_t> refp{0};
        tree ref_tree(refp);
        REQUIRE(test_tree_isomorphism(nt, ref_tree));

        REQUIRE((nm.size() == 1 && nm(0) == 2));
    }

    TEST_CASE("quasi flat zone hierarchy", "[hierarchy_core]") {

        auto graph = get_4_adjacency_graph({2, 3});

        array_1d<double> edge_weights{1, 0, 2, 1, 1, 1, 2};

        auto res = quasi_flat_zone_hierarchy(graph, edge_weights);
        auto rtree = res.tree;
        auto altitudes = res.altitudes;
        tree tref(array_1d<index_t>{6, 7, 8, 6, 7, 8, 7, 9, 9, 9});
        REQUIRE(test_tree_isomorphism(rtree, tref));
        REQUIRE(xt::allclose(altitudes, xt::xarray<double>({0, 0, 0, 0, 0, 0, 0, 1, 1, 2})));
    }

    TEST_CASE("saliency map", "[hierarchy_core]") {

        auto graph = get_4_adjacency_graph({2, 4});

        tree t(xt::xarray<long>{8, 8, 9, 9, 10, 10, 11, 11, 12, 13, 12, 14, 13, 14, 14});
        array_1d<double> altitudes{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3};

        auto sm = saliency_map(graph, t, altitudes);
        array_1d<double> sm_ref = {0, 1, 2, 1, 0, 3, 3, 0, 3, 0};

        REQUIRE((sm == sm_ref));
    }

    TEST_CASE("saliency maps of canonical bpt and qfz hierarchy are the same", "[hierarchy_core]") {

        index_t size = 25;
        auto graph = get_4_adjacency_graph({size, size});
        auto edge_weights = xt::eval(xt::random::randint<int>({num_edges(graph)}, 0, 25));

        auto bpt = bpt_canonical(graph, edge_weights);
        auto qfz = quasi_flat_zone_hierarchy(graph, edge_weights);

        auto sm_bpt = saliency_map(graph, bpt.tree, bpt.altitudes);
        auto sm_qfz = saliency_map(graph, qfz.tree, qfz.altitudes);

        REQUIRE((sm_bpt == sm_qfz));
    }

    TEST_CASE("tree_2_binary_tree", "[hierarchy_core]") {
        array_1d<index_t> parents{9, 9, 10, 10, 10, 10, 11, 11, 11, 12, 12, 12, 12};
        tree t(parents);

        auto res = tree_2_binary_tree(t);
        array_1d<index_t> exp_parents{9, 9, 10, 10, 11, 12, 13, 13, 14, 15, 11, 12, 15, 14, 16, 16, 16};
        array_1d<index_t> exp_node_map{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12};


        REQUIRE((res.tree.parents() == exp_parents));
        REQUIRE((res.node_map == exp_node_map));
    }

}