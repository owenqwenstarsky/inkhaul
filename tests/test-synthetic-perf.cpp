// GPU-only throughput benchmark without a real model
// it builds a synthetic BPE vocab (inkling2 pre-tokenizer) and a random inkling2_loop model with LongHaul experts,
// then times tokenization, prompt processing and generation on one GPU device

#include "common.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "llama-cpp.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

struct perf_params {
    uint32_t n_embd        = 1024;
    uint32_t n_head        = 8;
    uint32_t n_head_kv     = 2;
    uint32_t n_layer       = 8;    // physical layers: prelude 2, core 4 (run 3 times), coda 2
    uint32_t n_expert      = 64;
    uint32_t n_expert_used = 8;
    uint32_t n_ff_exp      = 512;
    uint32_t n_vocab       = 32768;
    size_t   cache_mib     = 256;
    uint32_t n_text_kib    = 24;   // size of the prompt text
    uint32_t n_gen         = 128;
    uint32_t n_ubatch      = 512;
    int      n_rep         = 3;
    std::string model_path;        // cached synthetic model, regenerated if missing
    bool     verbose       = false;
};

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

//
// synthetic text
//

static const char * SPECIAL_TOKENS[] = { "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>", "</tool_call>" };

// a mix of prose, numbers, code, non-ASCII words and chat markup, deterministic for a seed
static std::string make_text(size_t n_bytes, uint32_t seed) {
    std::mt19937 rng(0);
    static const char * syl[] = { "ka", "ri", "to", "men", "sa", "lo", "ve", "an", "the", "ing", "er", "on", "re", "st", "ou", "pro", "con", "de", "al", "ex", "tion", "ly", "un", "in", "be", "is", "or", "at" };
    static const char * uni[] = { "caf\xC3\xA9", "na\xC3\xAFve", "\xD0\xBF\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82", "\xCE\xBA\xCF\x8C\xCF\x83\xCE\xBC\xCE\xB5", "\xE4\xBD\xA0\xE5\xA5\xBD\xE4\xB8\x96\xE7\x95\x8C", "\xF0\x9F\x98\x80", "\xC3\x9C" "ber", "Stra\xC3\x9F" "e", "\xE3\x81\x93\xE3\x82\x93\xE3\x81\xAB\xE3\x81\xA1\xE3\x81\xAF" };
    static const char * code[] = { "    for (int i = 0; i < n; ++i) {\n        sum += x[i] * w[i];\n    }\n", "def forward(self, x):\n    return self.proj(x) + x\n", "if (ctx == nullptr) { return -1; }\n", "\tconst auto & v = map.at(\"key\");\n" };
    static const char * contr[] = { "'s", "'re", "'ll", "'ve", "'d", "'t", "'M" };

    // a Zipf-like lexicon
    std::vector<std::string> lex(4000);
    for (auto & w : lex) {
        const int n = 1 + rng() % 4;
        for (int i = 0; i < n; i++) {
            w += syl[rng() % (sizeof(syl)/sizeof(syl[0]))];
        }
    }
    std::vector<double> wts(lex.size());
    for (size_t i = 0; i < wts.size(); i++) {
        wts[i] = 1.0 / (i + 1);
    }
    std::discrete_distribution<size_t> pick(wts.begin(), wts.end());
    rng.seed(seed);

    std::string out;
    out.reserve(n_bytes + 256);
    int n_sent = 0;
    while (out.size() < n_bytes) {
        const uint32_t r = rng() % 100;
        if (r < 3) {
            out += "<|im_start|>user\n";
        } else if (r < 5) {
            out += "<|im_end|>\n";
        } else if (r < 9) {
            out += code[rng() % (sizeof(code)/sizeof(code[0]))];
        }
        const int n_words = 4 + rng() % 16;
        for (int i = 0; i < n_words; i++) {
            if (i > 0 || !out.empty()) {
                out += ' ';
            }
            const uint32_t k = rng() % 100;
            if (k < 4) {
                out += std::to_string(rng() % 1000000);
            } else if (k < 7) {
                out += uni[rng() % (sizeof(uni)/sizeof(uni[0]))];
            } else {
                std::string w = lex[pick(rng)];
                if (i == 0 || k < 12) {
                    w[0] = toupper(w[0]);
                }
                if (k == 12) {
                    std::transform(w.begin(), w.end(), w.begin(), ::toupper);
                }
                out += w;
                if (k == 13) {
                    out += contr[rng() % (sizeof(contr)/sizeof(contr[0]))];
                }
            }
            if (rng() % 10 == 0) {
                out += ",";
            }
        }
        out += (rng() % 5 == 0) ? "?" : ".";
        if (++n_sent % 6 == 0) {
            out += "\n\n";
        }
    }
    return out;
}

//
// synthetic BPE vocab
//

static std::string cpt_to_utf8(uint32_t c) {
    std::string s;
    if (c < 0x80) {
        s += (char) c;
    } else if (c < 0x800) {
        s += (char) (0xC0 | (c >> 6));
        s += (char) (0x80 | (c & 0x3F));
    } else {
        s += (char) (0xE0 | (c >> 12));
        s += (char) (0x80 | ((c >> 6) & 0x3F));
        s += (char) (0x80 | (c & 0x3F));
    }
    return s;
}

// GPT-2 byte to unicode map
static std::vector<std::string> byte_map() {
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; b++) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b);
    std::vector<std::string> res(256);
    std::vector<bool> used(256, false);
    for (int b : bs) {
        used[b] = true;
        res[b] = cpt_to_utf8(b);
    }
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (!used[b]) {
            res[b] = cpt_to_utf8(256 + n++);
        }
    }
    return res;
}

// rough GPT-4o style words: an optional leading space or symbol, then a run of letters (non-ASCII bytes count as letters)
static std::vector<std::string> rough_words(const std::string & text) {
    std::vector<std::string> res;
    size_t i = 0;
    auto is_letter = [](unsigned char c) { return isalpha(c) || c >= 0x80; };
    while (i < text.size()) {
        size_t j = i;
        const unsigned char c = text[i];
        if (!is_letter(c) && !isdigit(c) && c != '\n' && j + 1 < text.size() && is_letter(text[j + 1])) {
            j++;
        }
        if (j < text.size() && is_letter(text[j])) {
            while (j < text.size() && is_letter(text[j])) j++;
        } else if (isdigit(c)) {
            while (j < text.size() && isdigit(text[j]) && j - i < 3) j++;
        } else if (isspace(c)) {
            while (j < text.size() && isspace(text[j])) j++;
        } else {
            while (j < text.size() && !is_letter(text[j]) && !isdigit(text[j]) && !isspace(text[j])) j++;
        }
        res.push_back(text.substr(i, j - i));
        i = j;
    }
    return res;
}

struct synth_vocab {
    std::vector<std::string> tokens;
    std::vector<int32_t>     types;
    std::vector<std::string> merges;
};

// byte tokens, then the words of a training text built left to right by merges, then the special tokens
static synth_vocab make_vocab(uint32_t n_vocab) {
    const auto bmap = byte_map();
    synth_vocab v;
    std::unordered_map<std::string, int> ids;
    auto add = [&](const std::string & t, int32_t type) {
        if (ids.emplace(t, (int) v.tokens.size()).second) {
            v.tokens.push_back(t);
            v.types.push_back(type);
        }
    };
    for (int b = 0; b < 256; b++) {
        add(bmap[b], LLAMA_TOKEN_TYPE_NORMAL);
    }

    const uint32_t n_special = sizeof(SPECIAL_TOKENS)/sizeof(SPECIAL_TOKENS[0]);
    const std::string train = make_text(4u << 20, 1234);
    std::unordered_map<std::string, int> freq;
    for (const auto & w : rough_words(train)) {
        freq[w]++;
    }
    std::vector<std::pair<int, std::string>> sorted;
    for (auto & kv : freq) {
        sorted.emplace_back(kv.second, kv.first);
    }
    std::sort(sorted.begin(), sorted.end(), [](const auto & a, const auto & b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });

    for (const auto & fw : sorted) {
        if (v.tokens.size() + n_special >= n_vocab) {
            break;
        }
        std::string prev;
        for (size_t k = 0; k < fw.second.size(); k++) {
            const std::string & b = bmap[(unsigned char) fw.second[k]];
            if (k > 0) {
                const std::string cur = prev + b;
                if (!ids.count(cur)) {
                    if (v.tokens.size() + n_special >= n_vocab) {
                        break;
                    }
                    v.merges.push_back(prev + " " + b);
                    add(cur, LLAMA_TOKEN_TYPE_NORMAL);
                }
                prev = cur;
            } else {
                prev = b;
            }
        }
    }
    // filler tokens up to the vocab size
    for (uint32_t i = 0; v.tokens.size() + n_special < n_vocab; i++) {
        add("<|filler_" + std::to_string(i) + "|>", LLAMA_TOKEN_TYPE_UNUSED);
    }
    for (const char * s : SPECIAL_TOKENS) {
        add(s, LLAMA_TOKEN_TYPE_CONTROL);
    }
    return v;
}

//
// synthetic model
//

struct kv_writer {
    gguf_context * ctx;
    llm_arch       arch;
    std::string k(llm_kv key) const { return llama_arch_kv_name(arch, key); }
    void u32(llm_kv key, uint32_t v)        { gguf_set_val_u32 (ctx, k(key).c_str(), v); }
    void f32(llm_kv key, float v)           { gguf_set_val_f32 (ctx, k(key).c_str(), v); }
    void b  (llm_kv key, bool v)            { gguf_set_val_bool(ctx, k(key).c_str(), v); }
    void str(llm_kv key, const char * v)    { gguf_set_val_str (ctx, k(key).c_str(), v); }
};

static void add_tensor_type(gguf_context * ctx, const std::string & name, ggml_type type) {
    ggml_tensor t;
    memset(&t, 0, sizeof(t));
    t.type = type;
    ggml_set_name(&t, name.c_str());
    gguf_add_tensor(ctx, &t);
}

static gguf_context_ptr make_meta(const perf_params & p, const synth_vocab & v) {
    const llm_arch arch = LLM_ARCH_INKLING2_LOOP;
    gguf_context_ptr ret(gguf_init_empty());
    kv_writer w { ret.get(), arch };

    const uint32_t n_vocab        = v.tokens.size();
    const uint32_t n_vocab_padded = GGML_PAD(n_vocab, 256);

    w.str(LLM_KV_GENERAL_ARCHITECTURE, llama_arch_name(arch));
    w.u32(LLM_KV_CONTEXT_LENGTH,       32768);
    w.u32(LLM_KV_EMBEDDING_LENGTH,     p.n_embd);
    w.u32(LLM_KV_BLOCK_COUNT,          p.n_layer);
    w.u32(LLM_KV_FEED_FORWARD_LENGTH,  p.n_ff_exp);
    w.u32(LLM_KV_ATTENTION_HEAD_COUNT,    p.n_head);
    w.u32(LLM_KV_ATTENTION_HEAD_COUNT_KV, p.n_head_kv);
    w.f32(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    w.u32(LLM_KV_ATTENTION_SLIDING_WINDOW,    1024);

    w.u32(LLM_KV_ATTENTION_RELATIVE_LENGTH,   4);
    w.u32(LLM_KV_ATTENTION_RELATIVE_EXTENT,   16);
    w.f32(LLM_KV_ATTENTION_QK_CLIP,           0.25f);
    w.f32(LLM_KV_ATTENTION_RELATIVE_CLIP,     0.1f);
    w.u32(LLM_KV_ATTENTION_LOG_SCALE_FLOOR,   4);
    w.f32(LLM_KV_ATTENTION_LOG_SCALE_ALPHA,   0.1f);
    w.u32(LLM_KV_ATTN_RES_BLOCK_SIZE,         2);
    w.u32(LLM_KV_LOOP_START,                  2);
    w.u32(LLM_KV_LOOP_END,                    p.n_layer - 2);
    w.u32(LLM_KV_LOOP_COUNT,                  3);
    w.u32(LLM_KV_LOOP_GATE_RANK,              8);
    w.b  (LLM_KV_LOOP_VARYING,                false);
    w.u32(LLM_KV_SHORTCONV_L_CACHE,           4);
    w.u32(LLM_KV_VOCAB_SIZE,                  n_vocab_padded);
    w.u32(LLM_KV_VOCAB_SIZE_UNPADDED,         n_vocab);
    w.f32(LLM_KV_ACTIVATION_SITU_BETA,        4.0f);
    w.f32(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, 25.0f);
    w.f32(LLM_KV_EXPERT_WEIGHTS_SCALE,        2.0f);
    w.f32(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     30.0f);
    w.u32(LLM_KV_EXPERT_COUNT,                p.n_expert);
    w.u32(LLM_KV_EXPERT_USED_COUNT,           p.n_expert_used);
    w.u32(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  p.n_ff_exp);
    {
        std::vector<uint8_t> pattern(p.n_layer);
        for (uint32_t i = 0; i < p.n_layer; ++i) {
            pattern[i] = i % 4 != 3;
        }
        gguf_set_arr_data(ret.get(), w.k(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN).c_str(), GGUF_TYPE_BOOL, pattern.data(), pattern.size());
    }

    {
        std::vector<const char *> toks(n_vocab);
        std::vector<const char *> merges(v.merges.size());
        for (uint32_t i = 0; i < n_vocab; i++) {
            toks[i] = v.tokens[i].c_str();
        }
        for (size_t i = 0; i < v.merges.size(); i++) {
            merges[i] = v.merges[i].c_str();
        }
        w.str(LLM_KV_TOKENIZER_MODEL, "gpt2");
        w.str(LLM_KV_TOKENIZER_PRE,   "inkling2");
        gguf_set_arr_str (ret.get(), w.k(LLM_KV_TOKENIZER_LIST).c_str(), toks.data(), toks.size());
        gguf_set_arr_data(ret.get(), w.k(LLM_KV_TOKENIZER_TOKEN_TYPE).c_str(), GGUF_TYPE_INT32, v.types.data(), v.types.size());
        gguf_set_arr_str (ret.get(), w.k(LLM_KV_TOKENIZER_MERGES).c_str(), merges.data(), merges.size());
        w.u32(LLM_KV_TOKENIZER_EOS_ID, n_vocab - (uint32_t) (sizeof(SPECIAL_TOKENS)/sizeof(SPECIAL_TOKENS[0])) + 2);
    }

    // the large matrices in the types of a Q4_K file
    add_tensor_type(ret.get(), "token_embd.weight", GGML_TYPE_Q6_K);
    add_tensor_type(ret.get(), "output.weight",     GGML_TYPE_Q6_K);
    for (uint32_t il = 0; il < p.n_layer; il++) {
        for (const char * n : { "attn_q", "attn_k", "attn_v", "attn_output", "attn_gate" }) {
            add_tensor_type(ret.get(), "blk." + std::to_string(il) + "." + n + ".weight", GGML_TYPE_Q8_0);
        }
        add_tensor_type(ret.get(), "blk." + std::to_string(il) + ".ffn_gate_up_exps.weight", GGML_TYPE_Q4_K);
        add_tensor_type(ret.get(), "blk." + std::to_string(il) + ".ffn_down_exps.weight",    GGML_TYPE_Q4_K);
    }
    return ret;
}

// a pool of normal draws per type, each tensor copies a cyclic slice of it, so a large model fills fast
static std::map<ggml_type, std::vector<uint8_t>> g_pools;
static const int64_t POOL_N = 16 * 1024 * 1024;

static const std::vector<uint8_t> & get_pool(ggml_type type) {
    auto it = g_pools.find(type);
    if (it != g_pools.end()) {
        return it->second;
    }
    std::vector<float> f(POOL_N);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    for (float & x : f) {
        x = dist(gen);
    }
    std::vector<uint8_t> & pool = g_pools[type];
    pool.resize(ggml_row_size(type, POOL_N));
    if (type == GGML_TYPE_F32) {
        memcpy(pool.data(), f.data(), pool.size());
    } else if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(f.data(), (ggml_fp16_t *) pool.data(), POOL_N);
    } else {
        ggml_quantize_chunk(type, f.data(), pool.data(), 0, 1, POOL_N, nullptr);
    }
    return pool;
}

static void set_tensor_data(ggml_tensor * t, void *) {
    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> buf(nbytes);
    if (strstr(t->name, "exp_probs_b")) {
        float * d = (float *) buf.data();
        for (int64_t i = 0; i < ggml_nelements(t); i++) {
            d[i] = 0.0f;
        }
    } else if (strstr(t->name, "norm") || strstr(t->name, "rope") || strstr(t->name, "ffn.act")) {
        GGML_ASSERT(t->type == GGML_TYPE_F32);
        float * d = (float *) buf.data();
        for (int64_t i = 0; i < ggml_nelements(t); i++) {
            d[i] = 1.0f;
        }
    } else {
        const auto & pool = get_pool(t->type);
        const size_t bsz = ggml_type_size(t->type);
        size_t pos = (std::hash<std::string>{}(t->name) % (pool.size() / bsz)) * bsz;
        size_t i = 0;
        while (i < nbytes) {
            const size_t n = std::min(pool.size() - pos, nbytes - i);
            memcpy(buf.data() + i, pool.data() + pos, n);
            i += n;
            pos = 0;
        }
    }
    ggml_backend_tensor_set(t, buf.data(), 0, nbytes);
}

static bool silent_progress(float, void *) {
    return true;
}

static void make_model_file(const perf_params & p, const std::string & path) {
    const double t0 = now_s();
    const synth_vocab v = make_vocab(p.n_vocab);
    gguf_context_ptr meta = make_meta(p, v);

    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = silent_progress;
    ggml_backend_dev_t no_dev[] = { nullptr };
    mp.devices = no_dev;
    llama_model_ptr model(llama_model_init_from_user(meta.get(), set_tensor_data, nullptr, mp));
    GGML_ASSERT(model);
    llama_model_save_with_meta(model.get(), meta.get(), path.c_str());
    fprintf(stderr, "generated %s (%.1f MiB, %zu tokens, %zu merges) in %.1f s\n", path.c_str(),
        std::filesystem::file_size(path) / (1024.0*1024.0), v.tokens.size(), v.merges.size(), now_s() - t0);
}

//
// benchmark
//

struct perf_result {
    double t_tok = 0.0;
    double t_pp  = 0.0;
    double t_tg  = 0.0;
    int    n_tok = 0;
    int    n_gen = 0;
    std::vector<llama_token> gen;
};

static perf_result run_once(llama_context * ctx, const llama_vocab * vocab, const std::string & text, const perf_params & p) {
    perf_result r;

    double t0 = now_s();
    std::vector<llama_token> toks(text.size() + 16);
    const int n = llama_tokenize(vocab, text.c_str(), text.size(), toks.data(), toks.size(), false, true);
    GGML_ASSERT(n > 0);
    toks.resize(n);
    r.t_tok = now_s() - t0;
    r.n_tok = n;

    llama_memory_clear(llama_get_memory(ctx), true);
    llama_synchronize(ctx);

    t0 = now_s();
    const int n_batch = llama_n_batch(ctx);
    common_batch batch(ctx);
    for (int i = 0; i < n; i += n_batch) {
        const int nb = std::min(n_batch, n - i);
        batch.clear();
        for (int j = 0; j < nb; j++) {
            batch.add(toks[i + j], i + j, 0, i + j == n - 1);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            GGML_ABORT("decode failed");
        }
    }
    llama_synchronize(ctx);
    r.t_pp = now_s() - t0;

    llama_sampler_ptr smpl(llama_sampler_init_greedy());
    t0 = now_s();
    for (uint32_t i = 0; i < p.n_gen; i++) {
        llama_token tok = llama_sampler_sample(smpl.get(), ctx, -1);
        r.gen.push_back(tok);
        batch.clear();
        batch.add(tok, n + i, 0, true);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            GGML_ABORT("decode failed");
        }
    }
    llama_synchronize(ctx);
    r.t_tg  = now_s() - t0;
    r.n_gen = p.n_gen;
    return r;
}

static void usage(const char * argv0) {
    printf("usage: %s [options]\n", argv0);
    printf("  -m PATH     synthetic model file, generated if missing (default: <tmp>/test-synthetic-perf-<config>.gguf)\n");
    printf("  -c MIB      LongHaul cache size in MiB (default: 256)\n");
    printf("  -t KIB      prompt text size in KiB (default: 24)\n");
    printf("  -n N        tokens to generate (default: 128)\n");
    printf("  -ub N       ubatch size (default: 512)\n");
    printf("  -r N        repetitions, the median is reported (default: 3)\n");
    printf("  --experts N, --embd N, --layers N, --vocab N, --ff N  model shape\n");
    printf("  -v          print the llama.cpp logs\n");
}

int main(int argc, char ** argv) {
    perf_params p;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                usage(argv[0]);
                exit(1);
            }
            return argv[++i];
        };
        if      (a == "-m")        p.model_path = next();
        else if (a == "-c")        p.cache_mib  = std::stoul(next());
        else if (a == "-t")        p.n_text_kib = std::stoul(next());
        else if (a == "-n")        p.n_gen      = std::stoul(next());
        else if (a == "-ub")       p.n_ubatch   = std::stoul(next());
        else if (a == "-r")        p.n_rep      = std::stoi(next());
        else if (a == "--experts") p.n_expert   = std::stoul(next());
        else if (a == "--embd")    p.n_embd     = std::stoul(next());
        else if (a == "--layers")  p.n_layer    = std::stoul(next());
        else if (a == "--vocab")   p.n_vocab    = std::stoul(next());
        else if (a == "--ff")      p.n_ff_exp   = std::stoul(next());
        else if (a == "-v")        p.verbose    = true;
        else { usage(argv[0]); return a == "-h" ? 0 : 1; }
    }

    static bool verbose = p.verbose;
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_WARN || verbose) {
            fputs(text, stderr);
        }
    }, nullptr);
    llama_backend_init();

    ggml_backend_dev_t gpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu = dev;
            break;
        }
    }
    if (gpu == nullptr) {
        printf("no GPU device, skipping\n");
        return 0;
    }

    if (p.model_path.empty()) {
        char name[256];
        snprintf(name, sizeof(name), "test-synthetic-perf-v2-e%u-x%u-l%u-v%u-f%u.gguf", p.n_embd, p.n_expert, p.n_layer, p.n_vocab, p.n_ff_exp);
        p.model_path = (std::filesystem::temp_directory_path() / name).string();
    }
    if (!std::filesystem::exists(p.model_path)) {
        make_model_file(p, p.model_path);
    }

    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = silent_progress;
    ggml_backend_dev_t devs[] = { gpu, nullptr };
    mp.devices = devs;
    llama_model_ptr model(llama_model_load_from_file(p.model_path.c_str(), mp));
    GGML_ASSERT(model);
    const llama_vocab * vocab = llama_model_get_vocab(model.get());

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = 16384;
    cp.n_batch  = 2048;
    cp.n_ubatch = p.n_ubatch;
    cp.longhaul_cache_size = p.cache_mib * 1024 * 1024;
    cp.swa_full = false; // as in common
    cp.no_perf  = false;
    llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
    GGML_ASSERT(ctx);

    const std::string text = make_text((size_t) p.n_text_kib * 1024, 7);

    // warm up: shader compilation and the first fill of the cache
    {
        perf_params pw = p;
        pw.n_gen = 8;
        run_once(ctx.get(), vocab, text.substr(0, 4096), pw);
    }

    std::vector<perf_result> res;
    for (int i = 0; i < p.n_rep; i++) {
        res.push_back(run_once(ctx.get(), vocab, text, p));
    }
    auto med = [&](double perf_result::*f) {
        std::vector<double> v;
        for (const auto & r : res) {
            v.push_back(r.*f);
        }
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    const double t_tok = med(&perf_result::t_tok);
    const double t_pp  = med(&perf_result::t_pp);
    const double t_tg  = med(&perf_result::t_tg);
    const int n_tok = res[0].n_tok;

    uint64_t h = 1469598103934665603ull;
    for (llama_token t : res[0].gen) {
        h = (h ^ (uint64_t) t) * 1099511628211ull;
    }

    printf("model: %s, n_ubatch %u, cache %zu MiB, device %s\n", p.model_path.c_str(), llama_n_ubatch(ctx.get()), p.cache_mib, ggml_backend_dev_name(gpu));
    printf("tokenize : %8.2f ms  %6d tokens from %zu bytes (%.2f MB/s)\n", t_tok*1e3, n_tok, text.size(), text.size() / t_tok / 1e6);
    printf("prompt   : %8.2f ms  %8.1f t/s\n", t_pp*1e3, n_tok / t_pp);
    printf("generate : %8.2f ms  %8.1f t/s\n", t_tg*1e3, p.n_gen / t_tg);
    printf("total    : %8.2f ms\n", (t_tok + t_pp + t_tg)*1e3);
    printf("gen hash : %016" PRIx64 "\n", h);

    llama_backend_free();
    return 0;
}
