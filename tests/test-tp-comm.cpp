// checks the all-reduce between two RPC servers on threads of this process (127.0.0.1 only)
// and the events that the host reaches (ggml_backend_event_arm_host)
//
//   test-tp-comm                  all checks
//   test-tp-comm --timeout-child  (internal) a peer that stops answering, run in a child process

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-rpc.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

static ggml_backend_dev_t get_gpu(ggml_backend_dev_t skip = nullptr) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(dev);
        if ((type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) && dev != skip) {
            return dev;
        }
    }
    return nullptr;
}

// The two servers need a device each: the async all-reduce makes the GPU of each rank wait for the
// other rank, and two servers on one Metal device share one queue, so they would wait for each other
// forever. With one device the servers use the blocking path. Set GGML_METAL_DEVICES=2 on a Mac.
static void get_gpus(ggml_backend_dev_t gpus[2]) {
    gpus[0] = get_gpu();
    gpus[1] = gpus[0] != nullptr ? get_gpu(gpus[0]) : nullptr;
    if (gpus[1] == nullptr) {
        gpus[1] = gpus[0];
        setenv("GGML_RPC_COMM_SYNC", "1", 1);
        printf("one GPU: the servers use the blocking all-reduce (set GGML_METAL_DEVICES=2 for the async one)\n");
    }
}

// a graph that waits for an armed event must not run before the host reaches it
static bool test_host_events(ggml_backend_dev_t gpu) {
    ggml_backend_t backend = ggml_backend_dev_init(gpu, nullptr);
    GGML_ASSERT(backend != nullptr);
    ggml_backend_event_t done  = ggml_backend_event_new(gpu);
    ggml_backend_event_t after = ggml_backend_event_new(gpu);
    GGML_ASSERT(done != nullptr && after != nullptr);
    if (!ggml_backend_event_arm_host(done)) {
        printf("host events: SKIP (%s does not support them)\n", ggml_backend_dev_name(gpu));
        ggml_backend_event_free(done);
        ggml_backend_event_free(after);
        ggml_backend_free(backend);
        return true;
    }

    const int n = 1024;
    const ggml_init_params params = { 8*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr ctx(ggml_init(params));
    ggml_tensor * a = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor * b = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor * c = ggml_add(ctx.get(), a, b);
    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, c);
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    std::vector<float> va(n, 1.0f), vb(n, 2.0f), vc(n, 0.0f);
    ggml_backend_tensor_set(a, va.data(), 0, n*sizeof(float));
    ggml_backend_tensor_set(b, vb.data(), 0, n*sizeof(float));
    ggml_backend_tensor_set(c, vc.data(), 0, n*sizeof(float));

    bool ok = true;
    for (int round = 0; round < 3; round++) {
        if (round > 0) {
            GGML_ASSERT(ggml_backend_event_arm_host(done));
        }
        ggml_backend_event_wait(backend, done);
        GGML_ASSERT(ggml_backend_graph_compute_async(backend, gf) == GGML_STATUS_SUCCESS);
        ggml_backend_event_record(after, backend);

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (ggml_backend_event_query(after)) {
            printf("host events: FAIL (round %d: the graph ran before the host reached the event)\n", round);
            ok = false;
        }
        ggml_backend_event_signal_host(done);
        ggml_backend_event_synchronize(after);
        ggml_backend_tensor_get(c, vc.data(), 0, n*sizeof(float));
        if (vc[0] != 3.0f || vc[n - 1] != 3.0f) {
            printf("host events: FAIL (round %d: wrong result %f)\n", round, vc[0]);
            ok = false;
        }
        ggml_backend_synchronize(backend);
    }
    if (ok) {
        printf("host events: OK\n");
    }
    buf.reset();
    ggml_backend_event_free(done);
    ggml_backend_event_free(after);
    ggml_backend_free(backend);
    return ok;
}

struct rpc_pair {
    ggml_backend_t                       backends[2] = { nullptr, nullptr };
    void *                               comm        = nullptr;
    ggml_backend_comm_init_t             comm_init   = nullptr;
    ggml_backend_comm_free_t             comm_free   = nullptr;
    ggml_backend_comm_allreduce_tensor_t allreduce   = nullptr;
};

// two servers on ports p and p+1 of 127.0.0.1, one per GPU, and one backend on each
static bool start_pair(ggml_backend_dev_t gpus[2], rpc_pair & pair) {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("RPC");
    if (reg == nullptr) {
        return false;
    }
    auto start_server_fn = (decltype(ggml_backend_rpc_start_server) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_rpc_start_server");
    auto add_server_fn   = (decltype(ggml_backend_rpc_add_server) *)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_rpc_add_server");
    pair.comm_init = (ggml_backend_comm_init_t)             ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init");
    pair.comm_free = (ggml_backend_comm_free_t)             ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free");
    pair.allreduce = (ggml_backend_comm_allreduce_tensor_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor");
    GGML_ASSERT(start_server_fn && add_server_fn && pair.comm_init && pair.comm_free && pair.allreduce);

    std::random_device rd;
    const int port0 = 40000 + (int) (rd() % 18000);
    for (int i = 0; i < 2; i++) {
        const std::string endpoint = "127.0.0.1:" + std::to_string(port0 + i);
        ggml_backend_dev_t gpu = gpus[i];
        std::thread([start_server_fn, endpoint, gpu]() mutable {
            start_server_fn(endpoint.c_str(), nullptr, 4, 1, &gpu);
        }).detach();
        ggml_backend_reg_t reg_server = nullptr;
        for (int j = 0; j < 400 && reg_server == nullptr; j++) {
            reg_server = add_server_fn(endpoint.c_str());
            if (reg_server == nullptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
            }
        }
        if (reg_server == nullptr) {
            printf("the RPC server on %s did not start\n", endpoint.c_str());
            return false;
        }
        pair.backends[i] = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg_server, 0), nullptr);
        GGML_ASSERT(pair.backends[i] != nullptr);
    }
    pair.comm = pair.comm_init(pair.backends, 2);
    if (pair.comm == nullptr) {
        printf("the servers did not connect to each other\n");
        return false;
    }
    return true;
}

static void stop_pair(rpc_pair & pair) {
    if (pair.comm != nullptr) {
        pair.comm_free(pair.comm);
    }
    for (ggml_backend_t backend : pair.backends) {
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
}

// n_tensors partials of ne values per rank, all reduced back to back before any is read back
struct allreduce_case {
    const char * name;
    int64_t      ne;
    int          n_tensors;
    int          n_rounds;
    int          n_repeat;  // all-reduces of the same tensor in a row: the result doubles each time
};

static bool run_case(rpc_pair & pair, const allreduce_case & tc, std::mt19937 & rng) {
    const bool bf16 = tc.ne >= 32768;
    const ggml_init_params params = { (size_t) (tc.n_tensors + 1)*ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr ctx[2];
    ggml_backend_buffer_ptr buf[2];
    std::vector<ggml_tensor *> t[2];
    for (int r = 0; r < 2; r++) {
        ctx[r].reset(ggml_init(params));
        for (int j = 0; j < tc.n_tensors; j++) {
            ggml_tensor * x = ggml_new_tensor_1d(ctx[r].get(), GGML_TYPE_F32, tc.ne);
            x->flags |= GGML_TENSOR_FLAG_COMPUTE;
            t[r].push_back(x);
        }
        buf[r].reset(ggml_backend_alloc_ctx_tensors(ctx[r].get(), pair.backends[r]));
        GGML_ASSERT(buf[r] != nullptr);
    }

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<std::vector<float>> x[2];
    for (int r = 0; r < 2; r++) {
        x[r].assign(tc.n_tensors, std::vector<float>(tc.ne));
    }
    std::vector<float> got[2];
    got[0].resize(tc.ne);
    got[1].resize(tc.ne);

    double max_err = 0.0;
    for (int round = 0; round < tc.n_rounds; round++) {
        for (int r = 0; r < 2; r++) {
            for (int j = 0; j < tc.n_tensors; j++) {
                for (float & v : x[r][j]) {
                    v = dist(rng);
                }
                ggml_backend_tensor_set(t[r][j], x[r][j].data(), 0, tc.ne*sizeof(float));
            }
        }
        for (int j = 0; j < tc.n_tensors; j++) {
            for (int k = 0; k < tc.n_repeat; k++) {
                ggml_tensor * pair_t[2] = { t[0][j], t[1][j] };
                if (!pair.allreduce(pair.comm, pair_t)) {
                    printf("%s: FAIL (the all-reduce refused the tensors)\n", tc.name);
                    return false;
                }
            }
        }
        for (int j = 0; j < tc.n_tensors; j++) {
            ggml_backend_tensor_get(t[0][j], got[0].data(), 0, tc.ne*sizeof(float));
            ggml_backend_tensor_get(t[1][j], got[1].data(), 0, tc.ne*sizeof(float));
            // the mirrored result must be the same on both ranks, bit for bit
            if (memcmp(got[0].data(), got[1].data(), tc.ne*sizeof(float)) != 0) {
                printf("%s: FAIL (round %d, tensor %d: the ranks differ)\n", tc.name, round, j);
                return false;
            }
            const double scale = std::ldexp(1.0, tc.n_repeat - 1);
            for (int64_t i = 0; i < tc.ne; i++) {
                const double want = scale*((double) x[0][j][i] + (double) x[1][j][i]);
                max_err = std::max(max_err, std::fabs(got[0][i] - want) / scale);
            }
        }
    }
    // f32: one rounding of the sum; bf16: each partial rounded to 8 bits of mantissa
    const double tol = bf16 ? 2.0*std::ldexp(1.0, -8) : 1e-6;
    const bool ok = max_err <= tol;
    printf("%s: %s (max error %.2e)\n", tc.name, ok ? "OK" : "FAIL", max_err);
    return ok;
}

static bool test_allreduce(ggml_backend_dev_t gpus[2]) {
    rpc_pair pair;
    if (!start_pair(gpus, pair)) {
        printf("allreduce: FAIL (no pair of servers)\n");
        stop_pair(pair);
        return false;
    }
    std::mt19937 rng(42);
    const allreduce_case cases[] = {
        // decode-sized (n_embd 2048), up to 8 in flight: 10000 all-reduces
        { "decode 2048 x 8 in flight", 2048,    8, 1250, 1 },
        // the same tensor reduced 3 times in a row: each sum must see the one before
        { "decode 2048, 3 in a row",   2048,    4,   50, 3 },
        // prompt-sized (512 tokens x 2048): 4 MiB per rank, bf16 on the wire
        { "prompt 1M, bf16",           1 << 20, 2,   25, 1 },
        // the smallest bf16 payload
        { "bf16 threshold 32768",      32768,   4,   25, 1 },
    };
    bool ok = true;
    const auto t0 = std::chrono::steady_clock::now();
    for (const allreduce_case & tc : cases) {
        ok = run_case(pair, tc, rng) && ok;
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("allreduce: %s (%.1f s)\n", ok ? "OK" : "FAIL", s);
    stop_pair(pair);
    return ok;
}

// child: the peer stalls longer than the timeout, the run must stop with the error, not hang
static int timeout_child() {
    ggml_backend_load_all();
    ggml_backend_dev_t gpus[2];
    get_gpus(gpus);
    if (gpus[0] == nullptr) {
        return 0;
    }
    rpc_pair pair;
    if (!start_pair(gpus, pair)) {
        return 2;
    }
    const allreduce_case tc = { "stalled peer", 2048, 1, 1, 1 };
    std::mt19937 rng(1);
    run_case(pair, tc, rng);
    // not reached: reading the result fails on the server and the client aborts
    printf("the stalled peer did not stop the run\n");
    stop_pair(pair);
    return 3;
}

#ifndef _WIN32
static bool test_timeout(const char * argv0) {
    // rank 1 waits 2 s before each exchange, rank 0 gives up after 300 ms
    const std::string cmd = std::string("GGML_RPC_COMM_TIMEOUT_MS=300 GGML_RPC_COMM_TEST_DELAY=1:2000 '") + argv0 + "' --timeout-child 2>&1";
    const auto t0 = std::chrono::steady_clock::now();
    FILE * p = popen(cmd.c_str(), "r");
    if (p == nullptr) {
        printf("timeout: FAIL (cannot start the child)\n");
        return false;
    }
    std::string out;
    char line[4096];
    while (fgets(line, sizeof(line), p) != nullptr) {
        out += line;
    }
    const int status = pclose(p);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const bool failed_run = status != 0;
    const bool message    = out.find("failed or timed out") != std::string::npos;
    const bool ok = failed_run && message && s < 30.0;
    printf("timeout: %s (child status %d, %.1f s%s)\n", ok ? "OK" : "FAIL", status, s, message ? "" : ", no timeout message");
    if (!ok) {
        printf("%s\n", out.c_str());
    }
    return ok;
}
#endif

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "--timeout-child") == 0) {
        return timeout_child();
    }
    ggml_backend_load_all();
    ggml_backend_dev_t gpu = get_gpu();
    if (gpu == nullptr) {
        printf("test-tp-comm: SKIP (no GPU)\n");
        return 0;
    }
    bool ok = test_host_events(gpu);
    ggml_backend_dev_t gpus[2];
    get_gpus(gpus);
    ok = test_allreduce(gpus) && ok;
#ifndef _WIN32
    ok = test_timeout(argv[0]) && ok;
#endif
    printf("test-tp-comm: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
