// tp-probe: measures what decides whether tensor parallelism over a cable speeds up a token
//   listen / connect: TCP round trip and same-time exchange between two hosts
//   gpu:              the time to run and wait for a tiny graph on the local GPU
//   budget:           bytes read per token by part of a GGUF model, and the estimated speed of each mode
//   self-test:        the network code over 127.0.0.1 on two threads, and the budget of a tiny model

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

static int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double percentile(std::vector<double> v, double p) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    const size_t i = std::min(v.size() - 1, (size_t) (p * (v.size() - 1) + 0.5));
    return v[i];
}

//
// network
//

static void set_nodelay(int fd) {
    int flag = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
    int buf = 8*1024*1024;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof(buf));
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof(buf));
}

static bool send_all(int fd, const void * data, size_t size) {
    const char * p = (const char *) data;
    while (size > 0) {
        const ssize_t n = send(fd, p, size, 0);
        if (n <= 0) {
            return false;
        }
        p += n;
        size -= (size_t) n;
    }
    return true;
}

// spin: poll without sleeping, which saves the wake-up time of a blocking recv
static bool recv_all(int fd, void * data, size_t size, bool spin) {
    char * p = (char *) data;
    while (size > 0) {
        const ssize_t n = recv(fd, p, size, spin ? MSG_DONTWAIT : 0);
        if (n == 0) {
            return false;
        }
        if (n < 0) {
            if (spin && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                continue;
            }
            return false;
        }
        p += n;
        size -= (size_t) n;
    }
    return true;
}

// both ends send, then receive; a payload above the socket buffer is sent from a thread so that it cannot deadlock
static bool exchange(int fd, const std::vector<uint8_t> & out, std::vector<uint8_t> & in, bool spin) {
    if (out.size() <= 256*1024) {
        return send_all(fd, out.data(), out.size()) && recv_all(fd, in.data(), in.size(), spin);
    }
    std::atomic<bool> ok_send{true};
    std::thread sender([&]() { ok_send = send_all(fd, out.data(), out.size()); });
    const bool ok_recv = recv_all(fd, in.data(), in.size(), spin);
    sender.join();
    return ok_send && ok_recv;
}

enum probe_mode : uint32_t {
    PROBE_PING     = 1, // the listener echoes each message
    PROBE_EXCHANGE = 2, // both ends exchange at the same time
    PROBE_DONE     = 3,
};

struct probe_header {
    uint32_t mode;
    uint32_t spin;
    uint64_t size;
    uint64_t iters;
};

static void fill_pattern(std::vector<uint8_t> & buf, uint32_t seed) {
    for (size_t i = 0; i < buf.size(); ++i) {
        buf[i] = (uint8_t) ((i * 131 + seed) & 0xff);
    }
}

static bool check_pattern(const std::vector<uint8_t> & buf, uint32_t seed) {
    for (size_t i = 0; i < buf.size(); ++i) {
        if (buf[i] != (uint8_t) ((i * 131 + seed) & 0xff)) {
            return false;
        }
    }
    return true;
}

// serves one connection until the client sends PROBE_DONE; returns false on a protocol or data error
static bool serve(int fd) {
    probe_header h;
    while (recv_all(fd, &h, sizeof(h), false)) {
        if (h.mode == PROBE_DONE) {
            return true;
        }
        std::vector<uint8_t> buf(h.size);
        std::vector<uint8_t> out(h.size);
        for (uint64_t it = 0; it < h.iters; ++it) {
            if (h.mode == PROBE_PING) {
                if (!recv_all(fd, buf.data(), buf.size(), h.spin) || !send_all(fd, buf.data(), buf.size())) {
                    return false;
                }
            } else if (h.mode == PROBE_EXCHANGE) {
                fill_pattern(out, (uint32_t) (it + 1000));
                if (!exchange(fd, out, buf, h.spin) || !check_pattern(buf, (uint32_t) it)) {
                    return false;
                }
            } else {
                return false;
            }
        }
    }
    return false;
}

static int listen_on(const char * host, int port, int * port_out) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    int reuse = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t) port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1 || bind(fd, (sockaddr *) &addr, sizeof(addr)) != 0 || listen(fd, 1) != 0) {
        close(fd);
        return -1;
    }
    if (port_out) {
        socklen_t len = sizeof(addr);
        getsockname(fd, (sockaddr *) &addr, &len);
        *port_out = ntohs(addr.sin_port);
    }
    return fd;
}

static int connect_to(const char * host, int port) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t) port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1 || connect(fd, (sockaddr *) &addr, sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    set_nodelay(fd);
    return fd;
}

struct net_result {
    size_t size;
    bool   spin;
    double rtt_p50, rtt_p99;   // ping-pong round trip, us
    double xchg_p50, xchg_p99; // same-time exchange, us per round
};

static bool run_client(int fd, const std::vector<size_t> & sizes, int iters, bool spin, std::vector<net_result> & results) {
    for (size_t size : sizes) {
        net_result r = {};
        r.size = size;
        r.spin = spin;
        const int n = size > 1024*1024 ? std::max(10, iters/20) : iters;

        probe_header h = { PROBE_PING, spin ? 1u : 0u, size, (uint64_t) n };
        std::vector<uint8_t> out(size);
        std::vector<uint8_t> in(size);
        fill_pattern(out, 7);
        if (!send_all(fd, &h, sizeof(h))) {
            return false;
        }
        std::vector<double> t;
        for (int it = 0; it < n; ++it) {
            const int64_t t0 = now_us();
            if (!send_all(fd, out.data(), size) || !recv_all(fd, in.data(), size, spin)) {
                return false;
            }
            t.push_back((double) (now_us() - t0));
        }
        if (!check_pattern(in, 7)) {
            return false;
        }
        r.rtt_p50 = percentile(t, 0.5);
        r.rtt_p99 = percentile(t, 0.99);

        h.mode = PROBE_EXCHANGE;
        if (!send_all(fd, &h, sizeof(h))) {
            return false;
        }
        t.clear();
        for (int it = 0; it < n; ++it) {
            fill_pattern(out, (uint32_t) it);
            const int64_t t0 = now_us();
            if (!exchange(fd, out, in, spin)) {
                return false;
            }
            t.push_back((double) (now_us() - t0));
            if (!check_pattern(in, (uint32_t) (it + 1000))) {
                return false;
            }
        }
        r.xchg_p50 = percentile(t, 0.5);
        r.xchg_p99 = percentile(t, 0.99);
        results.push_back(r);
    }
    return true;
}

static bool send_done(int fd) {
    probe_header done = { PROBE_DONE, 0, 0, 0 };
    return send_all(fd, &done, sizeof(done));
}

static void print_net(const std::vector<net_result> & results) {
    printf("%10s %5s %12s %12s %12s %14s %14s\n", "size", "spin", "rtt p50 us", "rtt p99 us", "one-way us", "exchange p50", "exchange p99");
    for (const auto & r : results) {
        printf("%10zu %5s %12.1f %12.1f %12.1f %14.1f %14.1f\n", r.size, r.spin ? "yes" : "no", r.rtt_p50, r.rtt_p99, r.rtt_p50/2, r.xchg_p50, r.xchg_p99);
    }
    printf("\none-way = rtt p50 / 2. The cost of one sync after Phase 2 is about the 8 KiB exchange p50 plus the GPU event delay (tp-probe gpu).\n");
    printf("Prompt processing moves about n_embd * n_ubatch * 2 bytes per sync (2 MiB for n_embd 2048 and 512 tokens).\n");
}

static std::vector<size_t> parse_sizes(const std::string & s) {
    std::vector<size_t> res;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t end = s.find(',', pos);
        if (end == std::string::npos) {
            end = s.size();
        }
        std::string v = s.substr(pos, end - pos);
        size_t mul = 1;
        if (!v.empty() && (v.back() == 'K' || v.back() == 'k')) { mul = 1024; v.pop_back(); }
        else if (!v.empty() && (v.back() == 'M' || v.back() == 'm')) { mul = 1024*1024; v.pop_back(); }
        res.push_back((size_t) std::stoull(v) * mul);
        pos = end + 1;
    }
    return res;
}

//
// gpu
//

struct gpu_result {
    double compute_p50, compute_p99; // graph compute + synchronize, us
    double event_p50,   event_p99;   // event record + host wait after an async compute, us
    double turn_p50,    turn_p99;    // host reaches an event the GPU waits for -> graph -> host sees the GPU's event, us
    double awake_p50,   awake_p99;   // the same with the device kept awake (ggml_backend_dev_keep_awake), us
    double gpu_us;                   // GPU time of one graph, from back-to-back runs, us
};

// n_work: matrix-vector products of 4096 x 4096 (f16, 32 MiB) per graph, to keep the GPU busy like a layer
// wait_us: the host waits this long before it reaches each event, like the peer's partial on the cable
static bool run_gpu(int iters, int n_work, int wait_us, gpu_result & res, std::string & dev_name) {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(d);
        if ((type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) &&
                strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(d)), "RPC") != 0) {
            dev = d;
            break;
        }
    }
    if (dev == nullptr) {
        return false;
    }
    dev_name = ggml_backend_dev_name(dev);
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        return false;
    }

    ggml_init_params params = { (16 + 4*(size_t) n_work)*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);
    ggml_tensor * c = ggml_add(ctx, a, b);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4096);
    for (int i = 0; i < n_work; ++i) {
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 4096, 4096);
        x = ggml_scale(ctx, ggml_mul_mat(ctx, w, x), 1e-3f);
        ggml_build_forward_expand(gf, x);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        return false;
    }
    ggml_backend_buffer_clear(buf, 0);
    std::vector<float> host(2048, 1.0f);
    ggml_backend_tensor_set(a, host.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, host.data(), 0, ggml_nbytes(b));

    // warm up, then the GPU time of one graph when the queue never drains
    for (int it = 0; it < 20; ++it) {
        ggml_backend_graph_compute(backend, gf);
    }
    {
        const int64_t t0 = now_us();
        for (int it = 0; it < iters; ++it) {
            ggml_backend_graph_compute_async(backend, gf);
        }
        ggml_backend_synchronize(backend);
        res.gpu_us = (double) (now_us() - t0) / iters;
    }

    std::vector<double> t;
    for (int it = 0; it < iters; ++it) {
        const int64_t t0 = now_us();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        t.push_back((double) (now_us() - t0));
    }
    res.compute_p50 = percentile(t, 0.5);
    res.compute_p99 = percentile(t, 0.99);

    res.event_p50 = res.event_p99 = -1.0;
    ggml_backend_event_t ev = ggml_backend_event_new(dev);
    if (ev != nullptr) {
        t.clear();
        for (int it = 0; it < iters; ++it) {
            const int64_t t0 = now_us();
            ggml_backend_graph_compute_async(backend, gf);
            ggml_backend_event_record(ev, backend);
            ggml_backend_event_synchronize(ev);
            t.push_back((double) (now_us() - t0));
        }
        res.event_p50 = percentile(t, 0.5);
        res.event_p99 = percentile(t, 0.99);
        ggml_backend_event_free(ev);
    }

    // the GPU side of one sync of the async all-reduce: the queue never drains, the GPU waits at
    // an event that the host reaches, runs a tiny graph, and reaches an event that the host spins on;
    // then the same with the device kept awake, as the RPC server does during the all-reduces
    res.turn_p50 = res.turn_p99 = -1.0;
    res.awake_p50 = res.awake_p99 = -1.0;
    auto keep_awake = (ggml_backend_dev_keep_awake_t)
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_keep_awake");
    const int n_slots = 4;
    std::vector<ggml_backend_event_t> done(n_slots), ready(n_slots);
    bool host_events = true;
    for (int k = 0; k < n_slots; ++k) {
        done[k]  = ggml_backend_event_new(dev);
        ready[k] = ggml_backend_event_new(dev);
        host_events = host_events && done[k] != nullptr && ready[k] != nullptr;
    }
    auto queue_slot = [&](int k) {
        ggml_backend_event_arm_host(done[k]);
        ggml_backend_event_wait(backend, done[k]);
        ggml_backend_graph_compute_async(backend, gf);
        ggml_backend_event_record(ready[k], backend);
    };
    if (host_events && ggml_backend_event_arm_host(done[0])) {
        ggml_backend_event_signal_host(done[0]); // the probe arm above is reached right away
        for (int awake = 0; awake < (keep_awake != nullptr ? 2 : 1); ++awake) {
            for (int k = 0; k < n_slots; ++k) {
                queue_slot(k);
            }
            t.clear();
            for (int it = 0; it < iters; ++it) {
                const int k = it % n_slots;
                if (awake) {
                    keep_awake(dev, 100000);
                }
                const int64_t t_wait = now_us();
                while (now_us() - t_wait < wait_us) {
                }
                const int64_t t0 = now_us();
                ggml_backend_event_signal_host(done[k]);
                while (!ggml_backend_event_query(ready[k])) {
                }
                t.push_back((double) (now_us() - t0));
                queue_slot(k);
            }
            // let the queued slots run, so that nothing waits when the backend is freed
            for (int i = 0; i < n_slots; ++i) {
                ggml_backend_event_signal_host(done[(iters + i) % n_slots]);
            }
            ggml_backend_synchronize(backend);
            (awake ? res.awake_p50 : res.turn_p50) = percentile(t, 0.5);
            (awake ? res.awake_p99 : res.turn_p99) = percentile(t, 0.99);
        }
    }
    for (int k = 0; k < n_slots; ++k) {
        if (done[k] != nullptr) {
            ggml_backend_event_free(done[k]);
        }
        if (ready[k] != nullptr) {
            ggml_backend_event_free(ready[k]);
        }
    }

    ggml_backend_tensor_get(c, host.data(), 0, ggml_nbytes(c));
    const bool ok = host[0] == 2.0f && host[2047] == 2.0f;
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok;
}

//
// budget
//

struct budget {
    std::string arch;
    int64_t n_layer_passes = 0;          // layer passes per token
    std::map<std::string, double> bytes; // bytes read per generated token, by part
};

static uint32_t gguf_u32(const gguf_context * g, const std::string & key, uint32_t def) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) {
        return def;
    }
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(g, id);
        case GGUF_TYPE_INT32:  return (uint32_t) gguf_get_val_i32(g, id);
        case GGUF_TYPE_UINT64: return (uint32_t) gguf_get_val_u64(g, id);
        case GGUF_TYPE_ARRAY:
            if (gguf_get_arr_n(g, id) > 0 && gguf_get_arr_type(g, id) == GGUF_TYPE_UINT32) {
                return ((const uint32_t *) gguf_get_arr_data(g, id))[0];
            }
            return def;
        default: return def;
    }
}

// one generated token: experts by the share selected, embeddings by one row, the rest in full;
// the layers of an inkling2_loop core count once per pass
static bool compute_budget(const char * fname, budget & b) {
    ggml_context * meta = nullptr;
    gguf_init_params params = { true, &meta };
    gguf_context * g = gguf_init_from_file(fname, params);
    if (g == nullptr) {
        return false;
    }
    const int64_t arch_id = gguf_find_key(g, "general.architecture");
    b.arch = arch_id >= 0 ? gguf_get_val_str(g, arch_id) : "unknown";
    const std::string & a = b.arch;

    const uint32_t n_block   = gguf_u32(g, a + ".block_count", 0);
    const uint32_t n_nextn   = gguf_u32(g, a + ".nextn_predict_layers", 0);
    const uint32_t n_expert  = gguf_u32(g, a + ".expert_count", 0);
    const uint32_t n_used    = gguf_u32(g, a + ".expert_used_count", 0);
    const uint32_t loop_beg  = gguf_u32(g, a + ".loop.start", 0);
    const uint32_t loop_end  = gguf_u32(g, a + ".loop.end", 0);
    const uint32_t loop_n    = gguf_u32(g, a + ".loop.count", 1);
    const uint32_t n_layer   = n_block - n_nextn; // the MTP layers are not run

    auto passes = [&](uint32_t il) -> double {
        return (loop_end > loop_beg && il >= loop_beg && il < loop_end) ? (double) loop_n : 1.0;
    };
    b.n_layer_passes = 0;
    for (uint32_t il = 0; il < n_layer; ++il) {
        b.n_layer_passes += (int64_t) passes(il);
    }

    static const std::regex re_blk("blk\\.(\\d+)\\.(.*)");
    bool has_output = false;
    const ggml_tensor * tok_embd = nullptr;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        const std::string name = gguf_get_tensor_name(g, i);
        const ggml_tensor * t = ggml_get_tensor(meta, name.c_str());
        const double nbytes = (double) ggml_nbytes(t);
        std::smatch m;
        if (std::regex_match(name, m, re_blk)) {
            const uint32_t il = (uint32_t) std::stoul(m[1].str());
            if (il >= n_layer) {
                continue;
            }
            const std::string rest = m[2].str();
            const double p = passes(il);
            if (rest.find("_exps") != std::string::npos) {
                const double share = n_expert > 0 ? std::min(1.0, (double) n_used / n_expert) : 1.0;
                b.bytes["routed experts"] += nbytes * share * p;
            } else if (rest.find("_shexp") != std::string::npos) {
                b.bytes["shared experts"] += nbytes * p;
            } else if (rest.rfind("ffn_gate_inp", 0) == 0 || rest.rfind("exp_probs_b", 0) == 0 || rest.rfind("ffn_exp_probs_b", 0) == 0) {
                b.bytes["router"] += nbytes * p;
            } else if (rest.rfind("ffn_", 0) == 0) {
                b.bytes["dense ffn"] += nbytes * p;
            } else {
                b.bytes["attention and other"] += nbytes * p;
            }
        } else if (name == "output.weight") {
            has_output = true;
            b.bytes["output"] += nbytes;
        } else if (name == "token_embd.weight") {
            tok_embd = t;
        } else {
            b.bytes["norms and other"] += nbytes;
        }
    }
    if (tok_embd != nullptr) {
        b.bytes["embeddings"] += (double) ggml_nbytes(tok_embd) / (double) tok_embd->ne[1];
        if (!has_output) {
            b.bytes["output"] += (double) ggml_nbytes(tok_embd); // tied output reads the whole table
        }
    }
    gguf_free(g);
    ggml_free(meta);
    return true;
}

struct mode_estimate {
    const char * name;
    double split_bytes;    // read half on each laptop
    double mirrored_bytes; // read in full on each laptop
    int64_t n_sync;
};

static std::vector<mode_estimate> estimate_modes(const budget & b) {
    auto get = [&](const char * k) {
        auto it = b.bytes.find(k);
        return it == b.bytes.end() ? 0.0 : it->second;
    };
    const double experts  = get("routed experts") + get("shared experts") + get("dense ffn");
    const double attn     = get("attention and other");
    const double output   = get("output");
    const double mirrored = get("router") + get("embeddings") + get("norms and other");
    return {
        { "one laptop",               0.0,                      experts + attn + output + mirrored, 0 },
        { "tensor, attention split",  experts + attn + output,  mirrored,                           2*b.n_layer_passes },
        { "tensor, attention mirror", experts + output,         attn + mirrored,                    b.n_layer_passes },
    };
}

static void print_budget(const budget & b, double bw_gbs, const std::vector<double> & t_us) {
    double total = 0.0;
    for (const auto & kv : b.bytes) {
        total += kv.second;
    }
    printf("arch %s, %" PRId64 " layer passes per token\n\n", b.arch.c_str(), b.n_layer_passes);
    printf("%-22s %12s %8s\n", "part", "MiB/token", "share");
    for (const auto & kv : b.bytes) {
        printf("%-22s %12.1f %7.1f%%\n", kv.first.c_str(), kv.second/1048576.0, total > 0 ? 100.0*kv.second/total : 0.0);
    }
    printf("%-22s %12.1f\n\n", "total", total/1048576.0);

    const auto modes = estimate_modes(b);
    const double t1_ms = modes[0].mirrored_bytes / (bw_gbs*1e9) * 1e3;
    printf("generation of one stream at %.0f GB/s per laptop, all weights in memory (tok/s, speed-up):\n", bw_gbs);
    printf("%-26s %6s", "mode", "syncs");
    for (double t : t_us) {
        printf("  t=%4.0f us     ", t);
    }
    printf("  break-even t\n");
    for (const auto & m : modes) {
        const double base_ms = (m.split_bytes/2 + m.mirrored_bytes) / (bw_gbs*1e9) * 1e3;
        printf("%-26s %6" PRId64, m.name, m.n_sync);
        for (double t : t_us) {
            const double ms = base_ms + m.n_sync * t * 1e-3;
            printf("  %6.1f (%4.2fx)", 1e3/ms, t1_ms/ms);
        }
        if (m.n_sync > 0) {
            printf("  %8.0f us", std::max(0.0, (t1_ms - base_ms) * 1e3 / m.n_sync));
        }
        printf("\n");
    }
    printf("\nt is the cost of one sync: the 8 KiB exchange p50 of 'tp-probe connect' plus the event delay of 'tp-probe gpu'.\n");
    printf("These are estimates from the file sizes; measure with llama-bench -sm layer,expert,tensor.\n");
}

//
// self-test
//

static bool self_test_net() {
    int port = 0;
    const int lfd = listen_on("127.0.0.1", 0, &port);
    if (lfd < 0) {
        fprintf(stderr, "self-test: cannot listen on 127.0.0.1\n");
        return false;
    }
    std::atomic<bool> served{false};
    std::thread server([&]() {
        const int fd = accept(lfd, nullptr, nullptr);
        if (fd >= 0) {
            set_nodelay(fd);
            served = serve(fd);
            close(fd);
        }
    });
    const int fd = connect_to("127.0.0.1", port);
    std::vector<net_result> results;
    bool ok = fd >= 0;
    ok = ok && run_client(fd, { 8*1024, 4*1024*1024 }, 40, false, results);
    ok = ok && run_client(fd, { 8*1024 }, 40, true, results);
    ok = ok && send_done(fd);
    if (fd >= 0) {
        close(fd);
    }
    server.join();
    close(lfd);
    ok = ok && served && results.size() == 3;
    if (ok) {
        print_net(results);
    }
    return ok;
}

static bool self_test_budget() {
    char path[] = "/tmp/tp-probe-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) {
        return false;
    }
    close(fd);

    // a toy loop model: 4 layers, the core [1, 3) runs 3 times, 8 experts of which 2 are used
    ggml_init_params params = { 64*ggml_tensor_overhead() + 1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "toy");
    gguf_set_val_u32(g, "toy.block_count", 4);
    gguf_set_val_u32(g, "toy.expert_count", 8);
    gguf_set_val_u32(g, "toy.expert_used_count", 2);
    gguf_set_val_u32(g, "toy.loop.start", 1);
    gguf_set_val_u32(g, "toy.loop.end", 3);
    gguf_set_val_u32(g, "toy.loop.count", 3);
    auto add = [&](const std::string & name, int64_t ne0, int64_t ne1, int64_t ne2) {
        ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
        ggml_set_name(t, name.c_str());
        memset(t->data, 0, ggml_nbytes(t));
        gguf_add_tensor(g, t);
    };
    add("token_embd.weight", 16, 100, 1);
    add("output.weight", 16, 100, 1);
    add("output_norm.weight", 16, 1, 1);
    for (int il = 0; il < 4; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        add(p + "attn_q.weight", 16, 16, 1);
        add(p + "ffn_gate_inp.weight", 16, 8, 1);
        add(p + "ffn_down_exps.weight", 4, 16, 8);
    }
    const bool written = gguf_write_to_file(g, path, false);
    gguf_free(g);
    ggml_free(ctx);
    if (!written) {
        unlink(path);
        return false;
    }

    budget b;
    const bool ok = compute_budget(path, b);
    unlink(path);
    if (!ok) {
        return false;
    }
    // passes: 1 + 3 + 3 + 1 = 8; experts per pass: 4*16*8*4 bytes * 2/8 = 512
    const double exp_experts = 8 * 512.0;
    const double exp_attn    = 8 * 16*16*4.0;
    const double exp_router  = 8 * 16*8*4.0;
    bool pass = b.n_layer_passes == 8 &&
        b.bytes["routed experts"] == exp_experts &&
        b.bytes["attention and other"] == exp_attn &&
        b.bytes["router"] == exp_router &&
        b.bytes["output"] == 16*100*4.0 &&
        b.bytes["embeddings"] == 16*4.0 &&
        b.bytes["norms and other"] == 16*4.0;
    const auto modes = estimate_modes(b);
    pass = pass && modes.size() == 3 && modes[1].n_sync == 16 && modes[2].n_sync == 8;
    if (!pass) {
        fprintf(stderr, "self-test: budget mismatch (passes %" PRId64 ", experts %.0f, attention %.0f)\n",
                b.n_layer_passes, b.bytes["routed experts"], b.bytes["attention and other"]);
    }
    return pass;
}

//
// main
//

static void usage(const char * argv0) {
    printf("usage:\n");
    printf("  %s listen  -H ADDR [-p PORT]                         wait for one 'connect' and answer it\n", argv0);
    printf("  %s connect -H ADDR [-p PORT] [-s SIZES] [-n N]        measure the link (sizes like 8K,64K,2M)\n", argv0);
    printf("  %s gpu [-n N] [--work W] [--wait US]                measure the local GPU sync delay (W: 32 MiB products per graph,\n", argv0);
    printf("                                                        US: the host waits this long per sync, like the cable)\n");
    printf("  %s budget -m MODEL [--bw GB/s] [--t US,US,...]      bytes per token and the estimate per mode\n", argv0);
    printf("  %s self-test                                        check this tool over 127.0.0.1\n", argv0);
    printf("\nBind 'listen' to the Thunderbolt address only, never to 0.0.0.0.\n");
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }
    const std::string cmd = argv[1];
    std::string host = "127.0.0.1";
    std::string model;
    std::string sizes_str = "8K,64K,2M";
    std::string t_str = "50,100,300";
    int port = 50100;
    int iters = 1000;
    int n_work = 0;
    int wait_us = 0;
    double bw = 120.0;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", a.c_str());
                exit(1);
            }
            return argv[++i];
        };
        if (a == "-H") { host = next(); }
        else if (a == "-p") { port = std::stoi(next()); }
        else if (a == "-s") { sizes_str = next(); }
        else if (a == "-n") { iters = std::stoi(next()); }
        else if (a == "-m") { model = next(); }
        else if (a == "--bw") { bw = std::stod(next()); }
        else if (a == "--t") { t_str = next(); }
        else if (a == "--work") { n_work = std::stoi(next()); }
        else if (a == "--wait") { wait_us = std::stoi(next()); }
        else { usage(argv[0]); return 1; }
    }

    if (cmd == "listen") {
        const int lfd = listen_on(host.c_str(), port, nullptr);
        if (lfd < 0) {
            fprintf(stderr, "cannot listen on %s:%d\n", host.c_str(), port);
            return 1;
        }
        printf("listening on %s:%d\n", host.c_str(), port);
        const int fd = accept(lfd, nullptr, nullptr);
        close(lfd);
        if (fd < 0) {
            return 1;
        }
        set_nodelay(fd);
        const bool ok = serve(fd);
        close(fd);
        printf("%s\n", ok ? "done" : "connection error");
        return ok ? 0 : 1;
    }
    if (cmd == "connect") {
        const int fd = connect_to(host.c_str(), port);
        if (fd < 0) {
            fprintf(stderr, "cannot connect to %s:%d\n", host.c_str(), port);
            return 1;
        }
        // blocking and spinning receive, in one session
        std::vector<net_result> results;
        const auto sizes = parse_sizes(sizes_str);
        const bool ok = run_client(fd, sizes, iters, false, results) && run_client(fd, sizes, iters, true, results) && send_done(fd);
        close(fd);
        if (!ok) {
            fprintf(stderr, "link error\n");
            return 1;
        }
        print_net(results);
        return 0;
    }
    if (cmd == "gpu") {
        gpu_result r;
        std::string name;
        if (!run_gpu(iters, n_work, wait_us, r, name)) {
            fprintf(stderr, "no GPU or the GPU check failed\n");
            return 1;
        }
        printf("%s: graph of %d matrix products: GPU time %.1f us (back to back)\n", name.c_str(), n_work, r.gpu_us);
        printf("%s: graph compute + wait: p50 %.1f us, p99 %.1f us\n", name.c_str(), r.compute_p50, r.compute_p99);
        if (r.event_p50 >= 0) {
            printf("%s: async compute + event record + host wait: p50 %.1f us, p99 %.1f us\n", name.c_str(), r.event_p50, r.event_p99);
        }
        if (r.turn_p50 >= 0) {
            printf("%s: busy queue, host reaches an event -> graph -> host sees the GPU event: p50 %.1f us, p99 %.1f us\n",
                   name.c_str(), r.turn_p50, r.turn_p99);
            // back-to-back graphs can overlap on the GPU, so the difference is an upper bound; with --work 0 it is close
            printf("%s: one sync of the async all-reduce costs the GPU at most about %.1f us (turnaround - GPU time), plus the one-way time of the link\n",
                   name.c_str(), r.turn_p50 - r.gpu_us);
        }
        if (r.awake_p50 >= 0) {
            printf("%s: the same with the GPU kept awake: p50 %.1f us, p99 %.1f us, so about %.1f us per sync\n",
                   name.c_str(), r.awake_p50, r.awake_p99, r.awake_p50 - r.gpu_us);
        }
        return 0;
    }
    if (cmd == "budget") {
        if (model.empty()) {
            usage(argv[0]);
            return 1;
        }
        budget b;
        if (!compute_budget(model.c_str(), b)) {
            fprintf(stderr, "cannot read %s\n", model.c_str());
            return 1;
        }
        std::vector<double> t_us;
        for (size_t v : parse_sizes(t_str)) {
            t_us.push_back((double) v);
        }
        print_budget(b, bw, t_us);
        return 0;
    }
    if (cmd == "self-test") {
        const bool ok_net    = self_test_net();
        const bool ok_budget = self_test_budget();
        printf("tp-probe self-test: network %s, budget %s\n", ok_net ? "OK" : "FAIL", ok_budget ? "OK" : "FAIL");
        return ok_net && ok_budget ? 0 : 1;
    }
    usage(argv[0]);
    return 1;
}
