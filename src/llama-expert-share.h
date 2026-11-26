#pragma once

#include <cstdint>
#include <vector>

// LLAMA_SPLIT_MODE_EXPERT: which device computes each expert of a MoE layer

// the number of experts of each device: round(share*n_expert) by the largest remainder, so they sum to n_expert
// shares are >= 0 with a positive sum, they do not need to be normalized
std::vector<int32_t> llama_expert_share_counts(int32_t n_expert, const std::vector<double> & shares);

// the device of each expert, with the counts of llama_expert_share_counts spread evenly over the expert ids
std::vector<int32_t> llama_expert_share_owners(int32_t n_expert, const std::vector<double> & shares);
