#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define RPC_PROTO_MAJOR_VERSION    10
#define RPC_PROTO_MINOR_VERSION    5
#define RPC_PROTO_PATCH_VERSION    1

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 102, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");
#endif

#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);

GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(
        const char * endpoint,
        const char * cache_dir,
        size_t n_threads,
        size_t n_devices,
        ggml_backend_dev_t * devices);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

// server: the .gguf files in dir that clients may ask the server to read tensor data from (LongHaul expert misses)
// call before ggml_backend_rpc_start_server, returns the number of files
GGML_BACKEND_API size_t ggml_backend_rpc_server_set_model_dir(const char * dir, size_t n_io_threads);

// the size and fingerprint of a file, as the server and the client compute them to match their copies of a model file
GGML_BACKEND_API bool ggml_backend_rpc_file_fingerprint(const char * path, uint64_t * size, uint64_t * fingerprint);

// client: the id of the server's copy of the file at path, -1 if the server has no file with the same size and fingerprint
GGML_BACKEND_API int32_t ggml_backend_rpc_dev_open_file(ggml_backend_dev_t dev, const char * path);

struct ggml_backend_rpc_file_read {
    struct ggml_tensor * tensor;      // in a buffer of the server
    uint64_t             offset;      // in the tensor data
    int32_t              file;        // from ggml_backend_rpc_dev_open_file
    uint64_t             file_offset;
    uint64_t             size;
};

// client: the server reads the file ranges into the tensors, in order with the other commands to the server
GGML_BACKEND_API void ggml_backend_rpc_dev_set_tensors_from_file(ggml_backend_dev_t dev, const struct ggml_backend_rpc_file_read * reads, size_t n);

// server: tokenizes text with the vocabulary of the model file at path, as llama_tokenize does
// returns the number of tokens, minus the number needed if n_tokens_max is too small, INT32_MIN on error
typedef int32_t (*ggml_backend_rpc_tokenize_t)(const char * path, const char * text, size_t text_len,
        int32_t * tokens, int32_t n_tokens_max, bool add_special, bool parse_special);

// server: the tokenizer of the program that serves (ggml does not know the vocabulary), call before ggml_backend_rpc_start_server
GGML_BACKEND_API void ggml_backend_rpc_server_set_tokenizer(ggml_backend_rpc_tokenize_t fn);

// client: true if the server of dev can tokenize with its copies of the model files
GGML_BACKEND_API bool ggml_backend_rpc_dev_can_tokenize(ggml_backend_dev_t dev);

// client: tokenizes text on the server with its copy of a model file, named by size and fingerprint (ggml_backend_rpc_file_fingerprint)
// returns as llama_tokenize, INT32_MIN if the server has no such file
GGML_BACKEND_API int32_t ggml_backend_rpc_dev_tokenize(ggml_backend_dev_t dev, uint64_t file_size, uint64_t file_fingerprint,
        const char * text, size_t text_len, int32_t * tokens, int32_t n_tokens_max, bool add_special, bool parse_special);

#ifdef  __cplusplus
}
#endif
