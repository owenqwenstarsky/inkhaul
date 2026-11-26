// the experts of each device with LLAMA_SPLIT_MODE_EXPERT, see src/llama-expert-share.h

#include "../src/llama-expert-share.h"

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

int main() {
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
