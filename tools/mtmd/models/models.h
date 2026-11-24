#pragma once

#include "../mtmd-graph.h"

#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

/*
 * IMPORTANT: The mtmd module does NOT accept pull requests that are fully or predominantly AI-generated.
 * We encourage human contributors to ensure the quality and reliability of the codebase.
 */

struct mtmd_graph_inkling2v : mtmd_graph {
    using mtmd_graph::mtmd_graph;
    ggml_cgraph * build() override;
};
