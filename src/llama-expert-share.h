#pragma once

#include <cstdint>
#include <vector>

// LLAMA_SPLIT_MODE_EXPERT: which device computes each expert of a MoE layer

// the number of experts of each device: round(share*n_expert) by the largest remainder, so they sum to n_expert
// shares are >= 0 with a positive sum, they do not need to be normalized
std::vector<int32_t> llama_expert_share_counts(int32_t n_expert, const std::vector<double> & shares);

// the device of each expert, with the counts of llama_expert_share_counts spread evenly over the expert ids
std::vector<int32_t> llama_expert_share_owners(int32_t n_expert, const std::vector<double> & shares);

struct llama_expert_move {
    int32_t expert;
    int32_t from;
    int32_t to;
};

// moves experts between devices so that the share of the selections that each device computes gets closer to its target
// counts:  [n_expert] selections of each expert (any scale, e.g. decayed)
// owner:   [n_expert] device of each expert, the moves are applied to it
// targets: [n_dev] target share of each device, normalized
// caps:    [n_dev] the most experts a device can own (0: it cannot take experts)
// cached:  [n_expert] the expert is in the cache of its owner, such an expert is moved only if it fits clearly better
// tol:     no move while every device is within tol of its target (0.02 = 2 points)
// max_moves: the most moves returned, a swap is two moves
// each step takes the device furthest above its target and the device furthest below, and moves the expert whose share of
// the selections is closest to the smaller of the two gaps but at most 1.5 times it, so that both gaps get smaller
// if the device below owns its cap of experts, the step swaps two experts instead, so that the share moved (the difference) is as above
std::vector<llama_expert_move> llama_expert_share_rebalance(const std::vector<double> & counts, std::vector<int32_t> & owner,
        const std::vector<double> & targets, const std::vector<int32_t> & caps, const std::vector<bool> & cached, double tol, int max_moves);
