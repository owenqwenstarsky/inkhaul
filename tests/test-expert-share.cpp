// the experts of each device with LLAMA_SPLIT_MODE_EXPERT, see src/llama-expert-share.h

#include "../src/llama-expert-share.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

static int n_fail = 0;

static void check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        n_fail++;
    }
}

// the counts sum to n_expert and each is within 1 of its exact share
static void test_counts(int32_t n_expert, const std::vector<double> & shares) {
    const std::vector<int32_t> counts = llama_expert_share_counts(n_expert, shares);
    double sum_shares = 0.0;
    for (double s : shares) {
        sum_shares += s;
    }
    int32_t sum = 0;
    for (size_t d = 0; d < counts.size(); ++d) {
        sum += counts[d];
        check(std::fabs(counts[d] - n_expert*shares[d]/sum_shares) < 1.0, "a count is within 1 of its share");
        check(shares[d] > 0.0 || counts[d] == 0, "a device without a share owns no experts");
    }
    check(sum == n_expert, "the counts sum to n_expert");
}

// the owners match the counts, and every prefix of the ids is spread like the shares
static void test_owners(int32_t n_expert, const std::vector<double> & shares) {
    const std::vector<int32_t> counts = llama_expert_share_counts(n_expert, shares);
    const std::vector<int32_t> owners = llama_expert_share_owners(n_expert, shares);
    check((int32_t) owners.size() == n_expert, "one owner per expert");
    std::vector<int32_t> seen(shares.size(), 0);
    for (int32_t e = 0; e < n_expert; ++e) {
        check(owners[e] >= 0 && owners[e] < (int32_t) shares.size(), "the owner is a device");
        seen[owners[e]]++;
        for (size_t d = 0; d < shares.size(); ++d) {
            const double pace = (double) counts[d]*(e + 1)/n_expert;
            check(std::fabs(seen[d] - pace) <= 1.0, "the experts of a device are spread over the ids");
        }
    }
    for (size_t d = 0; d < shares.size(); ++d) {
        check(seen[d] == counts[d], "the owners match the counts");
    }
    check(owners == llama_expert_share_owners(n_expert, shares), "the owners are deterministic");
}

// the share of the counted selections of each device
static std::vector<double> loads(const std::vector<double> & counts, const std::vector<int32_t> & owner, size_t n_dev) {
    std::vector<double> res(n_dev, 0.0);
    double total = 0.0;
    for (size_t e = 0; e < counts.size(); ++e) {
        res[owner[e]] += counts[e];
        total += counts[e];
    }
    for (double & r : res) {
        r /= total;
    }
    return res;
}

static double max_gap(const std::vector<double> & load, const std::vector<double> & targets) {
    double res = 0.0;
    for (size_t d = 0; d < load.size(); ++d) {
        res = std::max(res, std::fabs(load[d] - targets[d]));
    }
    return res;
}

static double sum_gap(const std::vector<double> & load, const std::vector<double> & targets) {
    double res = 0.0;
    for (size_t d = 0; d < load.size(); ++d) {
        res += std::fabs(load[d] - targets[d]);
    }
    return res;
}

// Zipf-like counts: the expert of rank r gets 1/(r + 1), the ranks are spread over the ids by a fixed permutation
static std::vector<double> zipf(int32_t n_expert, int32_t step) {
    std::vector<double> res(n_expert);
    for (int32_t e = 0; e < n_expert; ++e) {
        res[e] = 1.0/((e*step) % n_expert + 1);
    }
    return res;
}

static std::vector<int32_t> n_owned(const std::vector<int32_t> & owner, size_t n_dev) {
    std::vector<int32_t> res(n_dev, 0);
    for (int32_t o : owner) {
        res[o]++;
    }
    return res;
}

// repeated checks on fixed skewed counts reach the targets within the tolerance, each move makes the gaps smaller,
// the caps and max_moves hold
static void test_rebalance_converges(int32_t n_expert, const std::vector<double> & targets, const std::vector<int32_t> & caps) {
    const double tol = 0.02;
    const std::vector<double> counts = zipf(n_expert, 37);
    std::vector<int32_t> owner = llama_expert_share_owners(n_expert, targets);
    const std::vector<bool> cached(n_expert, false);
    int n_checks = 0;
    for (; n_checks < 1000; ++n_checks) {
        const double gap_before = sum_gap(loads(counts, owner, targets.size()), targets);
        const std::vector<int32_t> owner_before = owner;
        const auto moves = llama_expert_share_rebalance(counts, owner, targets, caps, cached, tol, 2);
        check(moves.size() <= 2, "at most max_moves moves");
        for (const auto & mv : moves) {
            check(owner_before[mv.expert] == mv.from && owner[mv.expert] == mv.to && mv.from != mv.to, "a move matches the owners");
        }
        const std::vector<int32_t> own = n_owned(owner, targets.size());
        for (size_t d = 0; d < targets.size(); ++d) {
            check(own[d] <= std::max(caps[d], n_owned(owner_before, targets.size())[d]), "a device takes experts only up to its cap");
        }
        if (moves.empty()) {
            break;
        }
        check(sum_gap(loads(counts, owner, targets.size()), targets) < gap_before, "the moves make the gaps smaller");
    }
    check(n_checks < 1000, "the rebalance stops");
    check(max_gap(loads(counts, owner, targets.size()), targets) <= tol, "the rebalance reaches the targets within the tolerance");
}

static void test_rebalance() {
    test_rebalance_converges(128, { 0.6, 0.4 }, { 128, 128 });
    test_rebalance_converges(128, { 0.3, 0.7 }, { 128, 128 });
    test_rebalance_converges(256, { 0.2, 0.3, 0.5 }, { 256, 256, 256 });
    test_rebalance_converges(512, { 0.5, 0.5 }, { 512, 512 });

    // the targets are reached without exceeding caps a little above the initial counts (77 and 51)
    test_rebalance_converges(128, { 0.6, 0.4 }, { 90, 64 });
    // caps at the initial counts: only swaps
    test_rebalance_converges(128, { 0.6, 0.4 }, { 77, 51 });
    test_rebalance_converges(128, { 0.3, 0.7 }, { 38, 90 });
    test_rebalance_converges(256, { 0.2, 0.3, 0.5 }, { 51, 77, 128 });

    // a swap when the device below its target owns its cap: the counts stay
    {
        const std::vector<double> counts = { 0.40, 0.20, 0.25, 0.10, 0.05 };
        std::vector<int32_t> owner = { 0, 0, 1, 1, 1 };
        const auto moves = llama_expert_share_rebalance(counts, owner, { 0.5, 0.5 }, { 2, 3 }, std::vector<bool>(5, false), 0.02, 2);
        // device 0 is 10 points over: 0.20 against 0.10 moves exactly 10 points, the other pairs 15 points or more
        check(moves.size() == 2 && moves[0].expert == 1 && moves[0].to == 1 && moves[1].expert == 3 && moves[1].to == 0 &&
              n_owned(owner, 2) == std::vector<int32_t>({ 2, 3 }), "a swap at the cap");
        std::vector<int32_t> o = { 0, 0, 1, 1, 1 };
        check(llama_expert_share_rebalance(counts, o, { 0.5, 0.5 }, { 2, 3 }, std::vector<bool>(5, false), 0.02, 1).empty(), "a swap needs two moves");
    }

    // no move within the tolerance: even counts give each device about its share of the experts
    {
        const std::vector<double> targets = { 0.6, 0.4 };
        std::vector<int32_t> owner = llama_expert_share_owners(128, targets);
        const std::vector<int32_t> owner_init = owner;
        const auto moves = llama_expert_share_rebalance(std::vector<double>(128, 1.0), owner, targets, { 128, 128 }, std::vector<bool>(128, false), 0.02, 2);
        check(moves.empty() && owner == owner_init, "no move within the tolerance");
    }

    // no counts, no move
    {
        std::vector<int32_t> owner = llama_expert_share_owners(16, { 0.5, 0.5 });
        check(llama_expert_share_rebalance(std::vector<double>(16, 0.0), owner, { 0.5, 0.5 }, { 16, 16 }, std::vector<bool>(16, false), 0.02, 2).empty(), "no counts, no move");
    }

    // deterministic
    {
        const std::vector<double> counts = zipf(128, 37);
        std::vector<int32_t> owner_a = llama_expert_share_owners(128, { 0.6, 0.4 });
        std::vector<int32_t> owner_b = owner_a;
        const auto a = llama_expert_share_rebalance(counts, owner_a, { 0.6, 0.4 }, { 128, 128 }, std::vector<bool>(128, false), 0.02, 2);
        const auto b = llama_expert_share_rebalance(counts, owner_b, { 0.6, 0.4 }, { 128, 128 }, std::vector<bool>(128, false), 0.02, 2);
        bool same = a.size() == b.size() && owner_a == owner_b;
        for (size_t i = 0; same && i < a.size(); ++i) {
            same = a[i].expert == b[i].expert && a[i].from == b[i].from && a[i].to == b[i].to;
        }
        check(same && !a.empty(), "the rebalance is deterministic");
    }

    // a device with a cap of 0 (no cache banks) takes no experts
    {
        std::vector<int32_t> owner(16, 0);
        const auto moves = llama_expert_share_rebalance(std::vector<double>(16, 1.0), owner, { 0.5, 0.5 }, { 16, 0 }, std::vector<bool>(16, false), 0.02, 2);
        check(moves.empty(), "a device with a cap of 0 takes no experts");
    }

    // device 0 is 10 points over: of two experts that fit as well, the one not cached moves; a cached one that fits
    // much better moves before one that is not cached
    {
        const std::vector<double>  counts = { 0.30, 0.10, 0.10, 0.50 };
        const std::vector<int32_t> owner  = { 0, 0, 0, 1 };
        std::vector<int32_t> o = owner;
        auto moves = llama_expert_share_rebalance(counts, o, { 0.4, 0.6 }, { 4, 4 }, { false, true, false, false }, 0.02, 1);
        check(moves.size() == 1 && moves[0].expert == 2 && moves[0].to == 1, "an expert that is not cached moves first");

        const std::vector<double> counts2 = { 0.37, 0.10, 0.03, 0.50 };
        o = owner;
        moves = llama_expert_share_rebalance(counts2, o, { 0.4, 0.6 }, { 4, 4 }, { false, true, false, false }, 0.02, 1);
        check(moves.size() == 1 && moves[0].expert == 1, "a cached expert that fits much better moves");
    }

    // an expert larger than 1.5 times the gap does not move
    {
        std::vector<int32_t> owner = { 0, 1 };
        check(llama_expert_share_rebalance({ 0.6, 0.4 }, owner, { 0.5, 0.5 }, { 2, 2 }, { false, false }, 0.02, 2).empty(), "an expert larger than 1.5 times the gap does not move");
    }

    // the routing changes back and forth between two patterns, with the counts halved between checks as in the cache:
    // the owners settle instead of moving at every check
    {
        const std::vector<double> targets = { 0.6, 0.4 };
        const std::vector<double> a = zipf(128, 37);
        const std::vector<double> b = zipf(128, 91);
        std::vector<double> counts(128, 0.0);
        std::vector<int32_t> owner = llama_expert_share_owners(128, targets);
        int moves_early = 0;
        int moves_late  = 0;
        for (int i = 0; i < 200; ++i) {
            const std::vector<double> & add = i % 2 == 0 ? a : b;
            for (size_t e = 0; e < counts.size(); ++e) {
                counts[e] = 0.5*counts[e] + add[e];
            }
            const int n = (int) llama_expert_share_rebalance(counts, owner, targets, { 128, 128 }, std::vector<bool>(128, false), 0.02, 2).size();
            (i < 100 ? moves_early : moves_late) += n;
        }
        printf("alternating routing: %d moves in the first 100 checks, %d in the next 100\n", moves_early, moves_late);
        check(moves_late <= 10, "the owners settle under alternating routing");
    }

    // sizes that do not match throw
    {
        std::vector<int32_t> owner = { 0, 1 };
        bool thrown = false;
        try {
            llama_expert_share_rebalance({ 1.0 }, owner, { 0.5, 0.5 }, { 2, 2 }, { false, false }, 0.02, 2);
        } catch (const std::invalid_argument &) {
            thrown = true;
        }
        check(thrown, "sizes that do not match throw");
    }
}

int main() {
    test_rebalance();

    const std::vector<std::vector<double>> cases = {
        { 1.0 },
        { 1.0, 1.0 },
        { 30.0, 70.0 },
        { 60.0, 40.0 },
        { 100.0, 0.0 },
        { 0.0, 100.0 },
        { 1.0, 1.0, 1.0 },
        { 20.0, 30.0, 50.0 },
        { 0.1, 99.9 },
    };
    for (const int32_t n_expert : { 1, 2, 7, 16, 128, 256, 512 }) {
        for (const auto & shares : cases) {
            test_counts(n_expert, shares);
            test_owners(n_expert, shares);
        }
    }

    // 60/40 of 128 experts: 77 and 51 (76.8 rounds up)
    const std::vector<int32_t> c = llama_expert_share_counts(128, { 60.0, 40.0 });
    check(c.size() == 2 && c[0] == 77 && c[1] == 51, "60/40 of 128 is 77/51");

    // invalid shares
    for (const auto & shares : std::vector<std::vector<double>>{ {}, { 0.0, 0.0 }, { -1.0, 2.0 } }) {
        bool thrown = false;
        try {
            llama_expert_share_counts(16, shares);
        } catch (const std::invalid_argument &) {
            thrown = true;
        }
        check(thrown, "invalid shares throw");
    }

    if (n_fail > 0) {
        fprintf(stderr, "%d checks failed\n", n_fail);
        return 1;
    }
    printf("OK\n");
    return 0;
}
