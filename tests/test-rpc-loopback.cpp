#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-rpc.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

// an RPC server on a thread of this process, listening on 127.0.0.1 only:
// - partial writes to a weight buffer do not go through the server's hash cache (the LongHaul cache fills)
// - the server reads tensor data from its copy of a model file, and only from a matching copy

namespace fs = std::filesystem;

static void write_file(const fs::path & path, const std::vector<uint8_t> & data) {
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    ofs.write((const char *) data.data(), data.size());
    GGML_ASSERT(ofs.good());
}

static size_t count_files(const fs::path & dir) {
    size_t n = 0;
    for (const auto & entry : fs::directory_iterator(dir)) {
        n += entry.is_regular_file() ? 1 : 0;
    }
    return n;
}

int main(void) {
    ggml_backend_load_all();

    const fs::path dir = fs::temp_directory_path() / ("test-rpc-loopback-" + std::to_string(getpid()));
    const fs::path dir_cache  = dir / "cache";
    const fs::path dir_server = dir / "server";
    const fs::path dir_client = dir / "client";
    fs::create_directories(dir_cache);
    fs::create_directories(dir_server);
    fs::create_directories(dir_client);

    std::mt19937 rng(42);
    auto random_bytes = [&](size_t n) {
        std::vector<uint8_t> d(n);
        for (uint8_t & b : d) {
            b = (uint8_t) rng();
        }
        return d;
    };

    // the server and the client have the same model file, the client has another file of the same size
    const std::vector<uint8_t> model = random_bytes(20ull*1024*1024);
    write_file(dir_server / "model.gguf", model);
    write_file(dir_client / "model.gguf", model);
    write_file(dir_client / "other.gguf", random_bytes(model.size()));
    GGML_ASSERT(ggml_backend_rpc_server_set_model_dir(dir_server.string().c_str(), 2) == 1);

    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(cpu != nullptr);
    const std::string endpoint = "127.0.0.1:" + std::to_string(40000 + getpid() % 10000 + 2);
    const std::string cache    = dir_cache.string();
    std::thread([endpoint, cache, cpu]() mutable {
        ggml_backend_rpc_start_server(endpoint.c_str(), cache.c_str(), 1, 1, &cpu);
    }).detach();

    ggml_backend_reg_t reg = nullptr;
    for (int i = 0; i < 400 && reg == nullptr; i++) {
        reg = ggml_backend_rpc_add_server(endpoint.c_str());
        if (reg == nullptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
    }
    GGML_ASSERT(reg != nullptr && "the RPC server did not start");
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

    int n_fail = 0;
    auto check = [&](const char * name, bool ok) {
        printf("%s: %s\n", name, ok ? "OK" : "FAIL");
        n_fail += ok ? 0 : 1;
    };

    ggml_init_params params = {
        /* .mem_size   = */ 4*ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    ggml_tensor * w = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, 12ll*1024*1024); // above the hash threshold
    ggml_tensor * t = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, 17ll*1024*1024);
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
    GGML_ASSERT(buf != nullptr);
    ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_clear(buf.get(), 0);

    // a get waits for the server to handle the writes before it
    uint8_t probe[16];

    // the hash cache: a partial write is sent as data, a whole tensor is cached
    {
        const std::vector<uint8_t> data = random_bytes(ggml_nbytes(w));
        ggml_backend_tensor_set(w, data.data(), 0, ggml_nbytes(w) - 1024*1024);
        ggml_backend_tensor_get(w, probe, 0, sizeof(probe));
        check("partial write not cached", count_files(dir_cache) == 0);
        ggml_backend_tensor_set(w, data.data(), 0, ggml_nbytes(w));
        ggml_backend_tensor_get(w, probe, 0, sizeof(probe));
        check("whole tensor cached", count_files(dir_cache) == 1);
    }

    // reads from the server's model file
    {
        const int32_t id = ggml_backend_rpc_dev_open_file(dev, (dir_client / "model.gguf").string().c_str());
        check("open the matching file", id >= 0);
        check("open a file the server does not have", ggml_backend_rpc_dev_open_file(dev, (dir_client / "other.gguf").string().c_str()) == -1);
        if (id >= 0) {
            // two small reads and one of several chunks, which the server splits over its I/O threads
            const ggml_backend_rpc_file_read reads[] = {
                { t, 0,                id, 1000,            4096                },
                { t, 8192,             id, 2ull*1024*1024,  512*1024 + 3        },
                { t, 1ull*1024*1024,   id, 3ull*1024*1024,  16ull*1024*1024 - 5 },
            };
            ggml_backend_rpc_dev_set_tensors_from_file(dev, reads, 3);
            std::vector<uint8_t> out(ggml_nbytes(t));
            ggml_backend_tensor_get(t, out.data(), 0, out.size());

            std::vector<uint8_t> expected(ggml_nbytes(t), 0);
            for (const auto & r : reads) {
                memcpy(expected.data() + r.offset, model.data() + r.file_offset, r.size);
            }
            check("tensor data read from the file", out == expected);
        }
    }

    buf.reset();
    fs::remove_all(dir);
    return n_fail == 0 ? 0 : 1;
}
