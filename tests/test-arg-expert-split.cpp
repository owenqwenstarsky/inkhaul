// checks how -es, -kvd and -lhcd are resolved against the devices of this machine (no model is loaded)
// run with GGML_METAL_DEVICES=2 (or two GPUs) for the checks of the split itself

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "speculative.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

static bool parse(const std::vector<std::string> & args, common_params & params) {
    std::vector<std::string> argv = { "binary_name", "-m", "model_file.gguf" };
    argv.insert(argv.end(), args.begin(), args.end());
    std::vector<char *> ptrs;
    for (auto & arg : argv) {
        ptrs.push_back(const_cast<char *>(arg.data()));
    }
    params = common_params();
    return common_params_parse(ptrs.size(), ptrs.data(), params, LLAMA_EXAMPLE_COMMON);
}

static bool parse_ok(const std::vector<std::string> & args, common_params & params) {
    const bool ok = parse(args, params);
    if (!ok) {
        std::string s;
        for (const auto & a : args) {
            s += " " + a;
        }
        fprintf(stderr, "test-arg-expert-split: unexpected failure:%s\n", s.c_str());
    }
    return ok;
}

static bool parse_fails(const std::vector<std::string> & args) {
    common_params params;
    const bool ok = parse(args, params);
    if (ok) {
        std::string s;
        for (const auto & a : args) {
            s += " " + a;
        }
        fprintf(stderr, "test-arg-expert-split: unexpected success:%s\n", s.c_str());
    }
    return !ok;
}

static std::vector<ggml_backend_dev_t> devices_of(const common_params & params) {
    std::vector<ggml_backend_dev_t> res;
    for (ggml_backend_dev_t dev : params.devices) {
        if (dev != nullptr) {
            res.push_back(dev);
        }
    }
    return res;
}

static bool near(float a, float b) {
    return std::fabs(a - b) < 1e-3f;
}

int main(void) {
    common_params params;

    // -sm expert is a split mode like the others
    assert(parse_ok({ "-sm", "expert" }, params));
    assert(params.split_mode == LLAMA_SPLIT_MODE_EXPERT);
    assert(params.devices.empty());

    // the first local GPU, which the resolved device list must contain
    ggml_backend_load_all();
    std::string gpu0;
    for (size_t i = 0; i < ggml_backend_dev_count() && gpu0.empty(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            gpu0 = ggml_backend_dev_name(dev);
        }
    }
    if (gpu0.empty()) {
        printf("test-arg-expert-split: no GPU, skipping the device checks\n");
        return 0;
    }

    // -lhcd only: the device list is resolved, the split mode stays as it is
    assert(parse_ok({ "-lhcd", gpu0 + "=1G" }, params));
    const auto devs = devices_of(params);
    const size_t n = devs.size();
    assert(n >= 1);
    assert(params.devices.size() == n + 1 && params.devices.back() == nullptr);
    assert(params.split_mode == LLAMA_SPLIT_MODE_LAYER);
    assert(params.longhaul_cache_sizes.size() == llama_max_devices());
    size_t i0 = n;
    for (size_t i = 0; i < n; ++i) {
        if (gpu0 == ggml_backend_dev_name(devs[i])) {
            i0 = i;
        }
    }
    assert(i0 < n);
    for (size_t i = 0; i < params.longhaul_cache_sizes.size(); ++i) {
        assert(params.longhaul_cache_sizes[i] == (i == i0 ? 1ull << 30 : 0));
    }
    // the description also names a device
    assert(parse_ok({ "-lhcd", std::string(ggml_backend_dev_description(devs[i0])) + "=512M" }, params));
    assert(params.longhaul_cache_sizes[i0] == 512ull << 20);

    assert(parse_fails({ "-lhcd", gpu0 + "=0" }));
    assert(parse_fails({ "-lhcd", gpu0 }));
    assert(parse_fails({ "-lhcd", "NODEV=1G" }));

    // -kvd needs the expert split
    assert(parse_fails({ "-kvd", gpu0 }));
    // a single share is for the RPC devices, there are none here
    assert(parse_fails({ "-es", "60" }));
    // -es sets the split
    assert(parse_fails({ "-es", "100", "-ts", "1" }));
    assert(parse_fails({ "-es", gpu0 + "=100", "-sm", "row" }));
    assert(parse_fails({ "-es", gpu0 + "=100", "-sm", "expert" }));
    assert(parse_fails({ "-es", gpu0 + "=101" }));
    assert(parse_fails({ "-es", gpu0 + "=-1" }));
    assert(parse_fails({ "-es", gpu0 + "=abc" }));
    assert(parse_fails({ "-es", "NODEV=100" }));

    if (n < 2) {
        printf("test-arg-expert-split: %zu device, skipping the split checks (set GGML_METAL_DEVICES=2)\n", n);
        printf("test-arg-expert-split: OK\n");
        return 0;
    }

    const std::string d0 = ggml_backend_dev_name(devs[0]);
    const std::string d1 = ggml_backend_dev_name(devs[1]);

    // shares in the order of the devices
    std::string list = "60";
    for (size_t i = 1; i < n; ++i) {
        list += "," + std::to_string(40.0/(n - 1));
    }
    assert(parse_ok({ "-es", list }, params));
    assert(params.split_mode == LLAMA_SPLIT_MODE_EXPERT);
    assert(near(params.tensor_split[0], 60.0f));
    for (size_t i = 1; i < n; ++i) {
        assert(near(params.tensor_split[i], 40.0f/(n - 1)));
    }
    assert(params.tensor_split[n] == 0.0f);
    assert(params.main_gpu == 0);

    // a percent sign is allowed
    assert(parse_ok({ "-es", d0 + "=60%" }, params));
    assert(near(params.tensor_split[0], 60.0f));

    // named devices, the others split the rest
    assert(parse_ok({ "-es", d1 + "=30" }, params));
    assert(near(params.tensor_split[1], 30.0f));
    for (size_t i = 0; i < n; ++i) {
        if (i != 1) {
            assert(near(params.tensor_split[i], 70.0f/(n - 1)));
        }
    }

    // 0% is allowed: the device keeps the KV cache and the layers but no experts
    assert(parse_ok({ "-es", d0 + "=0", "-kvd", d0 }, params));
    assert(near(params.tensor_split[0], 0.0f));
    assert(params.main_gpu == 0);

    // -kvd picks the device of the layers and the KV cache
    assert(parse_ok({ "-es", list, "-kvd", d1 }, params));
    assert(params.main_gpu == 1);

    // the env vars work like the flags
    setenv("LLAMA_ARG_EXPERT_SHARE", (d1 + "=25").c_str(), 1);
    setenv("LLAMA_ARG_KV_DEVICE", d1.c_str(), 1);
    setenv("LLAMA_ARG_LONGHAUL_CACHE_DEVICE", (d0 + "=2G").c_str(), 1);
    assert(parse_ok({}, params));
    assert(params.split_mode == LLAMA_SPLIT_MODE_EXPERT);
    assert(near(params.tensor_split[1], 25.0f));
    assert(params.main_gpu == 1);
    assert(params.longhaul_cache_sizes[0] == 2ull << 30);
    unsetenv("LLAMA_ARG_EXPERT_SHARE");
    unsetenv("LLAMA_ARG_KV_DEVICE");
    unsetenv("LLAMA_ARG_LONGHAUL_CACHE_DEVICE");

    // with -dev, the shares follow the order of -dev
    assert(parse_ok({ "-dev", d1 + "," + d0, "-es", "70,30" }, params));
    assert(devices_of(params).size() == 2);
    assert(devices_of(params)[0] == devs[1]);
    assert(near(params.tensor_split[0], 70.0f));
    assert(near(params.tensor_split[1], 30.0f));

    // a draft model leaves the expert split, by default it goes on the KV device
    {
        assert(parse_ok({ "-es", list, "-kvd", d1, "-lhcd", d0 + "=1G" }, params));
        params.speculative.draft.mparams.path = "draft.gguf";
        assert(params.speculative.has_dft());
        const common_params draft = common_base_params_to_speculative(params);
        assert(draft.split_mode == LLAMA_SPLIT_MODE_LAYER);
        assert(devices_of(draft).size() == 1 && devices_of(draft)[0] == devs[1]);
        assert(draft.main_gpu == 0);
        assert(draft.tensor_split[0] == 0.0f);
        assert(draft.longhaul_cache_sizes.empty());
    }

    // invalid splits
    assert(parse_fails({ "-es", "100" }));                       // one share for several devices
    assert(parse_fails({ "-es", d0 + "=60," + d1 + "=60" }));    // over 100%
    assert(parse_fails({ "-es", d0 + "=60," + d0 + "=40" }));    // named twice
    assert(parse_fails({ "-es", d0 + "=60,40" }));               // mixed forms
    if (n == 2) {
        assert(parse_fails({ "-es", d0 + "=60," + d1 + "=30" })); // all named, not 100%
        assert(parse_fails({ "-es", "60,30" }));                  // not 100%
        assert(parse_fails({ "-es", "60,30,10" }));               // too many
    }
    // a prefix of several names is ambiguous
    size_t common = 0;
    while (common < d0.size() && common < d1.size() && d0[common] == d1[common]) {
        common++;
    }
    if (common > 0) {
        assert(parse_fails({ "-es", d0.substr(0, common) + "=50" }));
    }

    printf("test-arg-expert-split: OK\n");
    return 0;
}
