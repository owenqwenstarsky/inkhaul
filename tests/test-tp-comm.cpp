// checks the all-reduce between two RPC servers on threads of this process (127.0.0.1 only)
// and the events that the host reaches (ggml_backend_event_arm_host)
//
//   test-tp-comm                        all checks
//   test-tp-comm --timeout-child        (internal) a peer that stops answering, run in a child process
//   test-tp-comm --ids-child same|differ (internal) the all-reduce with the expert ids check, run in a child process

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
#include <mutex>
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
    ggml_backend_comm_allreduce_tensor_check_t allreduce_check = nullptr;
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
    pair.allreduce_check = (ggml_backend_comm_allreduce_tensor_check_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor_check");
    GGML_ASSERT(start_server_fn && add_server_fn && pair.comm_init && pair.comm_free && pair.allreduce && pair.allreduce_check);

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
    int          ids_check = 0; // 0: no check, 1: the ranks send the same expert ids, 2: rank 1 sends one other id
};

static bool run_case(rpc_pair & pair, const allreduce_case & tc, std::mt19937 & rng) {
    const bool bf16 = tc.ne >= 32768;
    const ggml_init_params params = { (size_t) (tc.n_tensors + 3)*ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr ctx[2];
    ggml_backend_buffer_ptr buf[2];
    std::vector<ggml_tensor *> t[2];
    // the expert ids: 4 of 16 for 8 tokens, a strided view like the top-k of the argsort in the model
    const int n_expert = 16, n_used = 4, n_tok = 8;
    ggml_tensor * ids_base[2] = { nullptr, nullptr };
    ggml_tensor * ids[2]      = { nullptr, nullptr };
    for (int r = 0; r < 2; r++) {
        ctx[r].reset(ggml_init(params));
        for (int j = 0; j < tc.n_tensors; j++) {
            ggml_tensor * x = ggml_new_tensor_1d(ctx[r].get(), GGML_TYPE_F32, tc.ne);
            x->flags |= GGML_TENSOR_FLAG_COMPUTE;
            t[r].push_back(x);
        }
        if (tc.ids_check != 0) {
            ids_base[r] = ggml_new_tensor_2d(ctx[r].get(), GGML_TYPE_I32, n_expert, n_tok);
        }
        buf[r].reset(ggml_backend_alloc_ctx_tensors(ctx[r].get(), pair.backends[r]));
        GGML_ASSERT(buf[r] != nullptr);
        if (tc.ids_check != 0) {
            std::vector<int32_t> v(n_expert*n_tok);
            for (size_t i = 0; i < v.size(); i++) {
                v[i] = (int32_t) (i % n_expert);
            }
            // one selected id differs on rank 1; the values after the first n_used of a row are not compared
            if (tc.ids_check == 2 && r == 1) {
                v[3*n_expert + 2] = 15;
            }
            v[5*n_expert + n_used] = 100 + r;
            ggml_backend_tensor_set(ids_base[r], v.data(), 0, v.size()*sizeof(int32_t));
            ids[r] = ggml_view_2d(ctx[r].get(), ids_base[r], n_used, n_tok, ids_base[r]->nb[1], 0);
            ids[r]->buffer = ids_base[r]->buffer;
            ggml_set_name(ids[r], "test_ids-7");
        }
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
                const bool accepted = tc.ids_check != 0 ? pair.allreduce_check(pair.comm, pair_t, ids) : pair.allreduce(pair.comm, pair_t);
                if (!accepted) {
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
        // with the hash of the same expert ids after the partial
        { "decode 2048, ids check",    2048,    4,  100, 1, 1 },
        { "prompt 1M, ids check",      1 << 20, 1,    5, 1, 1 },
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

// child: the all-reduce with the ids check; "differ" must stop the run with the error, "same" must pass
static int ids_child(const char * mode) {
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
    const bool differ = strcmp(mode, "differ") == 0;
    const allreduce_case tc = { "ids check", 2048, 2, 3, 1, differ ? 2 : 1 };
    std::mt19937 rng(3);
    const bool ok = run_case(pair, tc, rng);
    if (differ) {
        // not reached: reading the result fails on the server and the client aborts
        printf("different ids did not stop the run\n");
        return 3;
    }
    stop_pair(pair);
    return ok ? 0 : 4;
}

// the servers log from their own threads
struct log_capture {
    std::mutex        mutex;
    std::string       text;
    ggml_log_callback cb_old = nullptr;
    void *            ud_old = nullptr;

    void start() {
        ggml_log_get(&cb_old, &ud_old);
        ggml_log_set([](ggml_log_level level, const char * t, void * ud) {
            log_capture * lc = (log_capture *) ud;
            {
                std::lock_guard<std::mutex> lock(lc->mutex);
                lc->text.append(t);
            }
            if (lc->cb_old != nullptr) {
                lc->cb_old(level, t, lc->ud_old);
            }
        }, this);
    }
    void stop() {
        ggml_log_set(cb_old, ud_old);
    }
    bool has(const std::string & s) {
        std::lock_guard<std::mutex> lock(mutex);
        return text.find(s) != std::string::npos;
    }
};

// GGML_RPC_COMM_PORT moves the port that rank 0 listens on for the other ranks
static bool test_comm_port(ggml_backend_dev_t gpus[2]) {
    std::random_device rd;
    // the servers use 40000-58000 and +1000 for the default comm port
    const int port = 30000 + (int) (rd() % 9000);
    const std::string port_str = std::to_string(port);
    bool ok = true;
    std::string detail;

    log_capture log;
    log.start();
    setenv("GGML_RPC_COMM_PORT", port_str.c_str(), 1);
    {
        rpc_pair pair;
        if (!start_pair(gpus, pair)) {
            ok = false;
            detail = "the servers did not connect on the given port";
        } else {
            if (!log.has("port " + port_str + ")")) {
                ok = false;
                detail = "the comm did not use the given port";
            }
            std::mt19937 rng(7);
            const allreduce_case tc = { "comm port", 2048, 2, 10, 1 };
            ok = run_case(pair, tc, rng) && ok;
        }
        stop_pair(pair);
    }
    // a bad value must stop the comm with a message, not fall back to another port
    setenv("GGML_RPC_COMM_PORT", "70000", 1);
    {
        rpc_pair pair;
        if (start_pair(gpus, pair)) {
            ok = false;
            detail = "an invalid port was accepted";
        } else if (!log.has("invalid GGML_RPC_COMM_PORT")) {
            ok = false;
            detail = "no message for an invalid port";
        }
        stop_pair(pair);
    }
    unsetenv("GGML_RPC_COMM_PORT");
    log.stop();

    printf("comm port: %s%s%s\n", ok ? "OK" : "FAIL", detail.empty() ? "" : " - ", detail.c_str());
    return ok;
}

// child: the peer stalls longer than the timeout, the run must stop with the error, not hang
static int timeout_child() {
    ggml_backend_load_all();
    ggml_backend_dev_t gpus[2];
    get_gpus(gpus);
    if (gpus[0] == nullptr || gpus[1] == gpus[0]) {
        printf("timeout child: SKIP\n"); // the test delay is only on the async path, which needs two GPUs
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
static bool run_child(const std::string & cmd, std::string & out, int & status) {
    FILE * p = popen(cmd.c_str(), "r");
    if (p == nullptr) {
        return false;
    }
    char line[4096];
    while (fgets(line, sizeof(line), p) != nullptr) {
        out += line;
    }
    status = pclose(p);
    return true;
}

// the ranks compare the expert ids at each all-reduce, on the async and on the blocking path
static bool test_ids_check(const char * argv0) {
    struct child_case {
        const char * name;
        const char * env;
        const char * mode;
        bool         must_fail;
    };
    const child_case cases[] = {
        { "different ids, async",    "",                      "differ", true  },
        { "different ids, blocking", "GGML_RPC_COMM_SYNC=1 ", "differ", true  },
        { "same ids, blocking",      "GGML_RPC_COMM_SYNC=1 ", "same",   false },
    };
    bool all_ok = true;
    for (const child_case & cc : cases) {
        const std::string cmd = std::string(cc.env) + "GGML_RPC_COMM_TIMEOUT_MS=5000 '" + argv0 + "' --ids-child " + cc.mode + " 2>&1";
        std::string out;
        int status = 0;
        bool ok = run_child(cmd, out, status);
        const bool message = out.find("picked different experts in test_ids-7") != std::string::npos;
        if (cc.must_fail) {
            ok = ok && status != 0 && message;
        } else {
            ok = ok && status == 0 && !message;
        }
        printf("ids check, %s: %s (child status %d%s)\n", cc.name, ok ? "OK" : "FAIL", status,
               cc.must_fail && !message ? ", no mismatch message" : "");
        if (!ok) {
            printf("%s\n", out.c_str());
        }
        all_ok = all_ok && ok;
    }
    return all_ok;
}

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
    if (status == 0 && out.find("timeout child: SKIP") != std::string::npos) {
        printf("timeout: SKIP (needs two GPUs, GGML_METAL_DEVICES=2 gives two)\n");
        return true;
    }
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
    if (argc > 2 && strcmp(argv[1], "--ids-child") == 0) {
        return ids_child(argv[2]);
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
    ok = test_comm_port(gpus) && ok;
#ifndef _WIN32
    ok = test_ids_check(argv[0]) && ok;
    ok = test_timeout(argv[0]) && ok;
#endif
    printf("test-tp-comm: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
