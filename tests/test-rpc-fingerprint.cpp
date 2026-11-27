#include "ggml.h"
#include "ggml-rpc.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

// the fingerprint that an RPC server and its client compare to match their copies of a model file:
// the size, the first 4 MiB and 64 blocks of 4 KiB at evenly spaced offsets

static std::string tmp_path(const char * name) {
    return (std::filesystem::temp_directory_path() / (std::string("test-rpc-fingerprint-") + name)).string();
}

static void write_file(const std::string & path, const std::vector<uint8_t> & data) {
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    ofs.write((const char *) data.data(), data.size());
    GGML_ASSERT(ofs.good());
}

static uint64_t fingerprint(const std::string & path, uint64_t expected_size) {
    uint64_t size = 0;
    uint64_t fp   = 0;
    GGML_ASSERT(ggml_backend_rpc_file_fingerprint(path.c_str(), &size, &fp));
    GGML_ASSERT(size == expected_size);
    return fp;
}

int main(void) {
    const size_t size = 6ull*1024*1024;
    std::vector<uint8_t> data(size);
    std::mt19937 rng(42);
    for (uint8_t & b : data) {
        b = (uint8_t) rng();
    }

    const std::string path = tmp_path("a.gguf");
    write_file(path, data);
    const uint64_t fp = fingerprint(path, size);

    int n_fail = 0;
    auto check = [&](const char * name, bool ok) {
        printf("%s: %s\n", name, ok ? "OK" : "FAIL");
        n_fail += ok ? 0 : 1;
    };

    // the same content in another file
    {
        const std::string path_b = tmp_path("b.gguf");
        write_file(path_b, data);
        check("same content", fingerprint(path_b, size) == fp);
        std::filesystem::remove(path_b);
    }

    // one changed byte at an offset of the data, which the fingerprint must see or must not see
    auto changed = [&](size_t offset) {
        std::vector<uint8_t> d = data;
        d[offset] ^= 0xff;
        const std::string path_b = tmp_path("b.gguf");
        write_file(path_b, d);
        const uint64_t res = fingerprint(path_b, size);
        std::filesystem::remove(path_b);
        return res;
    };
    const size_t block_step = (size - 4096)/63;
    check("byte in the first 4 MiB",  changed(100) != fp);
    check("byte in the last block",   changed(size - 1) != fp);
    check("byte in block 50",         changed(50*(size - 4096)/63 + 10) != fp);
    // not a full hash: a byte between two sampled blocks after the first 4 MiB is not seen
    check("byte between blocks",      changed(50*(size - 4096)/63 + 4096 + block_step/2) == fp);

    // another size
    {
        std::vector<uint8_t> d(data.begin(), data.end() - 1);
        const std::string path_b = tmp_path("b.gguf");
        write_file(path_b, d);
        check("one byte shorter", fingerprint(path_b, size - 1) != fp);
        std::filesystem::remove(path_b);
    }

    // small and empty files, and a missing one
    {
        const std::string path_b = tmp_path("small.gguf");
        write_file(path_b, std::vector<uint8_t>(data.begin(), data.begin() + 100));
        const uint64_t fp_small = fingerprint(path_b, 100);
        write_file(path_b, {});
        const uint64_t fp_empty = fingerprint(path_b, 0);
        check("small vs. empty file", fp_small != fp_empty);
        std::filesystem::remove(path_b);

        uint64_t s  = 0;
        uint64_t f  = 0;
        check("missing file", !ggml_backend_rpc_file_fingerprint(tmp_path("missing.gguf").c_str(), &s, &f));
    }

    std::filesystem::remove(path);
    return n_fail == 0 ? 0 : 1;
}
