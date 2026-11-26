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
