#include "llama-expert-share.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

std::vector<int32_t> llama_expert_share_counts(int32_t n_expert, const std::vector<double> & shares) {
    const double sum = std::accumulate(shares.begin(), shares.end(), 0.0);
    if (n_expert < 0 || shares.empty() || !(sum > 0.0) || std::any_of(shares.begin(), shares.end(), [](double s) { return !(s >= 0.0); })) {
        throw std::invalid_argument("expert shares must be >= 0 with a positive sum");
    }
    std::vector<int32_t> res(shares.size());
    std::vector<double>  rem(shares.size());
    int32_t n = 0;
    for (size_t d = 0; d < shares.size(); ++d) {
        const double exact = n_expert*shares[d]/sum;
        res[d] = (int32_t) std::floor(exact);
        rem[d] = exact - res[d];
        n += res[d];
    }
    // the largest remainders get the rest, the lower device on a tie
    std::vector<size_t> order(shares.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return rem[a] > rem[b]; });
    for (size_t i = 0; n < n_expert; ++i) {
        res[order[i % order.size()]]++;
        n++;
    }
    return res;
}

std::vector<llama_expert_move> llama_expert_share_rebalance(const std::vector<double> & counts, std::vector<int32_t> & owner,
        const std::vector<double> & targets, const std::vector<int32_t> & caps, const std::vector<bool> & cached, double tol, int max_moves) {
    const size_t n_expert = counts.size();
    const size_t n_dev    = targets.size();
    if (owner.size() != n_expert || cached.size() != n_expert || caps.size() != n_dev) {
        throw std::invalid_argument("llama_expert_share_rebalance: sizes do not match");
    }
    std::vector<llama_expert_move> res;
    const double total = std::accumulate(counts.begin(), counts.end(), 0.0);
    if (!(total > 0.0)) {
        return res;
    }
    std::vector<double>  load(n_dev);
    std::vector<int32_t> n_own(n_dev);
    while (res.size() < (size_t) max_moves) {
        std::fill(load.begin(), load.end(), 0.0);
        std::fill(n_own.begin(), n_own.end(), 0);
        for (size_t e = 0; e < n_expert; ++e) {
            if (owner[e] < 0 || (size_t) owner[e] >= n_dev) {
                throw std::invalid_argument("llama_expert_share_rebalance: invalid owner");
            }
            load[owner[e]] += counts[e]/total;
            n_own[owner[e]]++;
        }

        // the device furthest above its target, and the one furthest below that can take experts, the lower device on a tie
        int32_t src = -1;
        int32_t dst = -1;
        for (size_t d = 0; d < n_dev; ++d) {
            const double diff = load[d] - targets[d];
            if (diff > 0.0 && (src < 0 || diff > load[src] - targets[src])) {
                src = (int32_t) d;
            }
            if (diff < 0.0 && caps[d] > 0 && (dst < 0 || diff < load[dst] - targets[dst])) {
                dst = (int32_t) d;
            }
        }
        if (src < 0 || dst < 0) {
            break;
        }
        const double excess  = load[src] - targets[src];
        const double deficit = targets[dst] - load[dst];
        if (excess <= tol && deficit <= tol) {
            break;
        }

        // a cached expert counts as a quarter of the gap further away, as its move costs a read on the new device
        const double want = std::min(excess, deficit);
        auto score_of = [&](double moved, size_t a, int32_t b) {
            if (!(moved > 0.0) || moved > 1.5*want) {
                return -1.0;
            }
            return std::fabs(moved - want) + (cached[a] ? 0.25*want : 0.0) + (b >= 0 && cached[b] ? 0.25*want : 0.0);
        };
        int32_t best   = -1;
        int32_t best_b = -1; // the expert of dst that goes back to src in a swap
        double  best_score = 0.0;
        const bool swap = n_own[dst] >= caps[dst];
        if (swap && res.size() + 2 > (size_t) max_moves) {
            break;
        }
        for (size_t a = 0; a < n_expert; ++a) {
            if (owner[a] != src) {
                continue;
            }
            const double fa = counts[a]/total;
            if (!swap) {
                const double score = score_of(fa, a, -1);
                if (score >= 0.0 && (best < 0 || score < best_score)) {
                    best       = (int32_t) a;
                    best_score = score;
                }
                continue;
            }
            for (size_t b = 0; b < n_expert; ++b) {
                if (owner[b] != dst) {
                    continue;
                }
                const double score = score_of(fa - counts[b]/total, a, (int32_t) b);
                if (score >= 0.0 && (best < 0 || score < best_score)) {
                    best       = (int32_t) a;
                    best_b     = (int32_t) b;
                    best_score = score;
                }
            }
        }
        if (best < 0) {
            break;
        }
        owner[best] = dst;
        res.push_back({ best, src, dst });
        if (best_b >= 0) {
            owner[best_b] = src;
            res.push_back({ best_b, dst, src });
        }
    }
    return res;
}

std::vector<int32_t> llama_expert_share_owners(int32_t n_expert, const std::vector<double> & shares) {
    const std::vector<int32_t> counts = llama_expert_share_counts(n_expert, shares);
    std::vector<int32_t> assigned(counts.size(), 0);
    std::vector<int32_t> res(n_expert);
    // expert e goes to the device that is furthest behind its even pace counts[d]*(e + 1)/n_expert, the lower device on a tie
    for (int32_t e = 0; e < n_expert; ++e) {
        int32_t best = -1;
        double best_lag = 0.0;
        for (size_t d = 0; d < counts.size(); ++d) {
            if (assigned[d] >= counts[d]) {
                continue;
            }
            const double lag = (double) counts[d]*(e + 1)/n_expert - assigned[d];
            if (best < 0 || lag > best_lag) {
                best     = (int32_t) d;
                best_lag = lag;
            }
        }
        res[e] = best;
        assigned[best]++;
    }
    return res;
}
