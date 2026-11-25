#include "llama-moe-cache.h"

#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-model.h"

#include "ggml-cpp.h"
#include "ggml-rpc.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

// ClassLRU: one LRU list per frequency class, over the experts of a group of layers
// slot_map[il][e] is the slot of expert e of layer il, n_slots or later if not cached, the host slot map tensor
// CRF (combined recency and frequency) = uses of an expert, decayed with a half-life of N tokens, retained across evictions
// class = floor(log2(CRF)), at most n_class - 1; each slot is in the LRU list of the class, as of its last use
// victim: an empty slot, or the least recently used slot of each list not in the current plan() with the max age/2^class
struct moe_cache_lru {
    static constexpr int   n_class  = 8; // floor(log2(1/(1 - 2^(-1/crf_half_life)))) = 7, class of CRF_max for half-life of 128
    static constexpr float crf_half_life = 128.0f; // tokens

    int32_t n_expert = 0;
    int32_t n_slots  = 0;
    int32_t empty    = 0; // slot of the experts that are not cached

    // [n_layer] data of the slot maps, the experts at slot n_slots or later are not cached
    std::vector<int32_t *> slot_map;
    std::vector<int32_t>   key_of;   // [n_slots] key of the expert in the slot, -1 if empty

    // one LRU list per class, the head is the least recently used end
    std::vector<int32_t> prev;    // [n_slots] -1 at the ends
    std::vector<int32_t> next;    // [n_slots]
    int32_t head[n_class];
    int32_t tail[n_class];
    std::vector<int8_t>  list_of; // [n_slots] class list of the slot
    int32_t n_filled = 0;         // slots that hold an expert

    std::vector<float>   crf;      // [n_layer*n_expert]
    std::vector<int64_t> last_use; // [n_layer*n_expert] layer clock at the last use, -1 if not used
    std::vector<int64_t> clock;    // [n_layer] tokens seen by the layer (as added by plan() for each ubatch)

    std::vector<uint32_t> seen; // [n_expert] seen[e] == seen_gen if expert e is selected by the current plan()
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;  // the distinct selected experts
    std::vector<int32_t> misses;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        GGML_ASSERT(n_class == (int) std::log2(1.0f/(1.0f - std::exp2(-1.0f/crf_half_life))) + 1);
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        this->empty    = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        for (int c = 0; c < n_class; ++c) {
            head[c] = -1;
            tail[c] = -1;
        }
        head[0] = 0;
        tail[0] = n_slots - 1;
        list_of.assign(n_slots, 0);
        crf.assign((size_t) n_layer*n_expert, 0.0f);
        last_use.assign((size_t) n_layer*n_expert, -1);
        clock.assign(n_layer, 0);
        seen.assign(n_expert, 0);
    }

    // move slot s to the most recently used end of list c
    void move(int32_t s, int c) {
        const int from = list_of[s];
        if (from == c && s == tail[from]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }

        prev[s] = tail[c];
        next[s] = -1;
        if (tail[c] >= 0) {
            next[tail[c]] = s;
        } else {
            head[c] = s;
        }
        tail[c] = s;
        list_of[s] = c;
    }

    // key = index of an expert of a layer in crf and last_use
    int32_t make_key(int32_t il, int32_t e) const { return il*n_expert + e; }
    int32_t key_layer (int32_t key) const { return key / n_expert; }
    int32_t key_expert(int32_t key) const { return key % n_expert; }

    // for a key after use(): its CRF is >= 1, so log2 is >= 0 and the cast is the floor
    int class_of(int32_t key) const {
        return std::min(n_class - 1, (int) std::log2(crf[key]));
    }

    // the CRF of a key, decayed by the time since its last use
    float decayed_crf(int32_t il, int32_t key) const {
        return crf[key]*std::exp2(-(clock[il] - last_use[key])/crf_half_life);
    }

    // an unused key gets 1 because crf[key] is 0 prior to first use
    void use(int32_t il, int32_t key) {
        crf[key]      = decayed_crf(il, key) + 1.0f;
        last_use[key] = clock[il];
    }

    // the experts selected by this plan() must not be evicted, nor the slots this plan() filled already
    bool is_selected(int32_t il, int32_t key) const {
        if (key < 0) {
            return false; // unused slot
        }
        if (key_layer(key) != il) {
            return false; // seen is per expert index, an expert of another layer can have the same index
        }
        return seen[key_expert(key)] == seen_gen;
    }

    // the least recently used slot of list c that this plan() does not select, -1 if none
    int32_t candidate(int c, int32_t il) const {
        for (int32_t s = head[c]; s >= 0; s = next[s]) {
            if (!is_selected(il, key_of[s])) {
                return s;
            }
        }
        return -1;
    }

    int32_t victim(int32_t il) const {
        int32_t res = -1;
        double res_score = -1.0;
        for (int c = 0; c < n_class; ++c) {
            const int32_t s = candidate(c, il);
            if (s < 0) {
                continue;
            }
            const int32_t key = key_of[s];
            if (key < 0) {
                return s; // unused slot
            }
            const int64_t age   = clock[key_layer(key)] - last_use[key];
            const double  score = std::ldexp((double) age, -c); // age/2^class
            if (score > res_score) {
                res       = s;
                res_score = score;
            }
        }
        if (res < 0) {
            GGML_ABORT("no MoE cache victim");
        }
        return res;
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids in layer il, at the clock of layer il
    // advance: move the clock of layer il by n_tokens (false for a later loop pass over the same experts)
    // returns false if the ids select more distinct experts than there are slots
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, int64_t n_tokens, bool advance, std::vector<fill> & fills, size_t & n_hit) {
        fills.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // update the CRF of all selected experts and promote hits before choosing victims for misses
        misses.clear();
        for (int32_t e : uniq) {
            const int32_t key = make_key(il, e);
            use(il, key);
            if (slots[e] < n_slots) {
                move(slots[e], class_of(key));
                n_hit++;
            } else {
                misses.push_back(e);
            }
        }

        // fill the misses in expert order: victim() takes unused slots, and slots filled by one plan(), in list order,
        // so consecutive experts often get consecutive slots and one upload
        std::sort(misses.begin(), misses.end());
        for (int32_t e : misses) {
            const int32_t s = victim(il);
            if (key_of[s] >= 0) {
                slot_map[key_layer(key_of[s])][key_expert(key_of[s])] = empty;
            } else {
                n_filled++;
            }
            const int32_t key = make_key(il, e);
            key_of[s] = key;
            slots[e]  = s;
            move(s, class_of(key));
            fills.push_back({ e, s });
        }
        if (advance) {
            clock[il] += n_tokens;
        }
        return true;
    }
};

// persistent threads for the disk reads of the cache misses, so that the SSD sees several reads at once
// run(n_jobs, fn) calls fn(job, thread) for each job, on the calling thread (0) and the pool threads (1..n_threads - 1)
class moe_cache_io_pool {
public:
    explicit moe_cache_io_pool(int n_threads) {
        for (int i = 1; i < n_threads; ++i) {
            threads.emplace_back([this, i] { worker(i); });
        }
    }

    ~moe_cache_io_pool() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stop = true;
        }
        cv_start.notify_all();
        for (auto & t : threads) {
            t.join();
        }
    }

    int n_threads() const { return (int) threads.size() + 1; }

    // rethrows the first exception of a job, the jobs not started yet are skipped
    void run(size_t n_jobs, const std::function<void(size_t, int)> & fn) {
        if (threads.empty() || n_jobs <= 1) {
            for (size_t j = 0; j < n_jobs; ++j) {
                fn(j, 0);
            }
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mtx);
            job_fn       = &fn;
            this->n_jobs = n_jobs;
            next         = 0;
            error        = nullptr;
            n_busy       = (int) threads.size();
            gen++;
        }
        cv_start.notify_all();
        work(0);
        std::exception_ptr err;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv_done.wait(lock, [&] { return n_busy == 0; });
            job_fn = nullptr;
            err    = error;
        }
        if (err) {
            std::rethrow_exception(err);
        }
    }

private:
    void work(int id) {
        for (size_t j = next.fetch_add(1); j < n_jobs; j = next.fetch_add(1)) {
            try {
                (*job_fn)(j, id);
            } catch (...) {
                std::lock_guard<std::mutex> lock(mtx);
                if (!error) {
                    error = std::current_exception();
                }
                next = n_jobs;
            }
        }
    }

    void worker(int id) {
        uint64_t seen = 0;
        std::unique_lock<std::mutex> lock(mtx);
        while (true) {
            cv_start.wait(lock, [&] { return stop || gen != seen; });
            if (stop) {
                return;
            }
            seen = gen;
            lock.unlock();
            work(id);
            lock.lock();
            if (--n_busy == 0) {
                cv_done.notify_one();
            }
        }
    }

    std::vector<std::thread> threads;
    std::mutex               mtx;
    std::condition_variable  cv_start;
    std::condition_variable  cv_done;

    // the current run, set under mtx before the threads start
    const std::function<void(size_t, int)> * job_fn = nullptr;
    size_t              n_jobs = 0;
    std::atomic<size_t> next   = 0;
    std::exception_ptr  error;
    uint64_t            gen    = 0;
    int                 n_busy = 0;
    bool                stop   = false;
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // experts on disk of the layers it caches
        double split = 0.0;    // share of the budget
        int32_t n_ubatch_reserved = 0; // tokens per ubatch that every group can hold before the rest is shared

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;

        // an RPC server with copies of the model files reads the missing experts itself
        std::vector<int32_t> server_files; // [model file] id of the server's copy, -1 if not needed, empty if the server does not read
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;      // with the loop passes that share the experts of another layer
        int32_t n_owners = 0;             // layers with their own experts
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_lru lru;
    };

    struct layer {
        int32_t ig    = -1; // -1 if the layer is not cached
        int32_t owner = -1; // layer that owns the experts, its LRU keys and slot map data are used (a loop pass reuses the experts of an earlier layer)
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        std::vector<ggml_tensor *> experts; // expert tensors on disk, in the order of the banks
    };

    struct stats {
        size_t  hits    = 0;
        size_t  misses  = 0;
        size_t  bytes   = 0;
        size_t  bytes_server = 0; // read by RPC servers from their copies of the model files
        int64_t read_us = 0;      // of the local reads
    };

    // reads go through host buffers, the device banks are not host memory
    // each I/O thread has its own staging buffer of this size, or of the largest expert if larger
    static constexpr size_t staging_size = 8ull*1024*1024;
    static constexpr int    n_io_threads_default = 4;

    // a read of consecutive experts of one tensor into consecutive slots
    struct read_job {
        const ggml_tensor * w;
        ggml_tensor *       bank;
        size_t              offs_src;
        size_t              offs_dst;
        size_t              size;
    };

    const llama_model & model;
    int32_t n_expert_used;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> bindings; // experts on disk -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> group_of;       // experts on disk -> group
    std::unordered_map<const ggml_tensor *, int32_t> layer_of;       // slot map -> layer

    std::vector<int32_t> ids;
    std::vector<moe_cache_lru::fill> fills;
    std::vector<read_job> jobs;
    std::vector<uint8_t *> staging; // [n_io_threads]
    size_t staging_thread_size = 0;
    ggml_backend_buffer_ptr staging_buf; // pinned memory of the device if it has a host buffer type (CUDA)
    std::vector<uint8_t> staging_mem;   // else plain memory (Metal)
    std::unique_ptr<moe_cache_io_pool> io_pool;
    std::mutex set_mtx; // the uploads to the banks, the backends do not promise concurrent set_tensor

    decltype(ggml_backend_rpc_dev_set_tensors_from_file) * rpc_set_from_file = nullptr;
    std::vector<ggml_backend_rpc_file_read> server_reads;

    // LLAMA_MOE_CACHE_TRACE: the selected experts of each slot lookup, for a simulator of the cache
    FILE * trace = nullptr;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    impl(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size) :
            model(model), n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()) {
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("LongHaul requires a GPU backend (CUDA, Metal or RPC) for the expert cache");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("LongHaul requires a MoE model");
        }

        // cache the layers whose experts are on disk, on the device the layer is assigned to
        std::unordered_map<const ggml_tensor *, int32_t> owner_of;
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            const size_t n_disk = std::count_if(experts.begin(), experts.end(), [&](const ggml_tensor * t) { return model.is_longhaul(t); });
            if (n_disk == 0) {
                continue;
            }
            if (n_disk != experts.size()) {
                throw std::runtime_error(format("LongHaul: layer %zu has experts both on disk and in memory", il));
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                throw std::runtime_error(format("LongHaul: layer %zu is not on the GPU, offload all layers", il));
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());

            // loop passes share the expert tensors of an earlier layer, they share its slots too
            int32_t owner = (int32_t) il;
            if (const auto it_own = owner_of.find(experts[0]); it_own != owner_of.end()) {
                owner = it_own->second;
                GGML_ASSERT(layers[owner].experts == experts);
                // the slots of the owner are in the banks of its device
                if (model.dev_layer(owner) != model.dev_layer(il)) {
                    throw std::runtime_error(format("LongHaul: layer %zu uses the experts of layer %d but is on device %s instead of %s",
                        il, owner, ggml_backend_dev_name(model.dev_layer(il)), ggml_backend_dev_name(model.dev_layer(owner))));
                }
            } else {
                for (const ggml_tensor * t : experts) {
                    owner_of[t] = owner;
                }
            }
            layers[il].owner   = owner;
            layers[il].experts = experts;

            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            if (owner == (int32_t) il) {
                it->n_owners++;
                for (const ggml_tensor * t : experts) {
                    it->host_bytes         += ggml_nbytes(t);
                    devices[id].host_bytes += ggml_nbytes(t);
                }
            }
        }
        if (groups.empty()) {
            throw std::runtime_error("LongHaul: the model has no experts on disk");
        }

        // one extra slot at the end, some kernels read past the last expert
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        if (const char * path = getenv("LLAMA_MOE_CACHE_TRACE"); path != nullptr && *path != '\0') {
            trace = fopen(path, "wb");
            if (trace == nullptr) {
                throw std::runtime_error(std::string("failed to open the MoE cache trace ") + path);
            }
            write_trace_header(split_sum);
        }
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;

        // within a device, every group first gets the slots of the same ubatch: the slots are reused layer by layer, so the
        // need of a group does not depend on its layer count, and it stops at the experts of one layer (see max_ubatch).
        // The largest ubatch that fits in half of the budget is reserved, the rest is split by the size of the experts, so
        // each group caches the same fraction of its experts on top. With one group the whole budget goes to it either way.
        auto max_slots = [&](const group & g) {
            return g.n_owners*(int32_t) g.ref[0]->ne[2];
        };
        auto reserved_slots = [&](const group & g, int32_t n_ubatch) {
            return (int32_t) std::min<int64_t>((int64_t) n_ubatch*n_expert_used, g.ref[0]->ne[2]);
        };
        auto reserve = [&](int32_t id, int32_t n_ubatch) {
            size_t res = 0;
            for (const group & g : groups) {
                if (g.id == id) {
                    res += alloc_size(g, reserved_slots(g, n_ubatch));
                }
            }
            return res;
        };
        size_t min_size = 0; // smallest budget that gives every group n_expert_used slots
        bool too_small = false;
        for (size_t id = 0; id < devices.size(); ++id) {
            device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            const double dev_share = d.split/split_sum;
            const size_t dev_size  = (size_t) ((double) size*dev_share);

            const size_t reserve_min = reserve(id, 1);
            min_size  = std::max(min_size, (size_t) std::ceil(reserve_min/dev_share));
            too_small = too_small || reserve_min > dev_size;

            // the reserve stops growing once every group holds the experts of one layer
            int32_t n_max = 1;
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    n_max = std::max(n_max, (int32_t) ((g.ref[0]->ne[2] + n_expert_used - 1)/n_expert_used));
                }
            }
            d.n_ubatch_reserved = 1;
            while (d.n_ubatch_reserved < n_max && reserve(id, d.n_ubatch_reserved + 1) <= dev_size/2) {
                d.n_ubatch_reserved++;
            }

            const size_t rest = dev_size - std::min(dev_size, reserve(id, d.n_ubatch_reserved));
            for (group & g : groups) {
                if (g.id != (int32_t) id) {
                    continue;
                }
                g.n_slots = reserved_slots(g, d.n_ubatch_reserved);
                const size_t budget = alloc_size(g, g.n_slots) + (size_t) ((double) rest*g.host_bytes/d.host_bytes);
                while (g.n_slots < max_slots(g) && alloc_size(g, g.n_slots + 1) <= budget) {
                    g.n_slots++;
                }
            }
        }
        if (too_small) {
            throw std::runtime_error(format("the LongHaul cache of %.0f MiB is too small to hold the experts of one token, use at least %.0f MiB",
                size/1024.0/1024.0, std::ceil(min_size/1024.0/1024.0)));
        }
        for (group & g : groups) {
            g.lru.init(model.layers.size(), g.ref[0]->ne[2], std::max(g.n_slots, 1));
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            n_tensors_host  += g.layers.size();
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host = init_ctx(n_tensors_host);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        size_t max_expert_size = 0;
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
                max_expert_size = std::max(max_expert_size, (size_t) t->nb[2]);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig = (int32_t) ig;
                if (l.owner == il) {
                    for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                        ggml_tensor * bank   = g.banks[ip];
                        ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots, bank->nb[1], bank->nb[2], 0);
                        ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                        bindings[l.experts[ip]] = cached;
                        group_of[l.experts[ip]] = (int32_t) ig;
                    }
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        // with no_alloc, dummy buffers are attached so the memory use can still be measured, see llama_context::memory_breakdown
        for (device & d : devices) {
            if (!d.ctx) {
                continue;
            }
            d.buf = llama_memory_bufs::alloc_buf(d.ctx.get(), d.buft, no_alloc, "MoE cache");
            if (!no_alloc) {
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
        }
        buf_host = llama_memory_bufs::alloc_buf(ctx_host.get(), buft_host, no_alloc, "MoE cache slot maps");
        if (!no_alloc) {
            ggml_backend_buffer_clear(buf_host.get(), 0);
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].owner == il) {
                        int32_t * slots = (int32_t *) layers[il].slot_map->data;
                        std::fill(slots, slots + layers[il].slot_map->ne[1], g.lru.empty);
                        g.lru.slot_map[il] = slots;
                    }
                }
            }

            // LLAMA_LONGHAUL_IO_THREADS: the number of concurrent disk reads, 1 reads on the compute thread only
            int n_io_threads = n_io_threads_default;
            if (const char * env = getenv("LLAMA_LONGHAUL_IO_THREADS"); env != nullptr && *env != '\0') {
                n_io_threads = std::clamp(atoi(env), 1, 64);
            }
            io_pool = std::make_unique<moe_cache_io_pool>(n_io_threads);

            // pinned memory makes the uploads to a discrete GPU faster, the buffers are shared by all devices
            staging_thread_size = GGML_PAD(std::max(staging_size, max_expert_size), 64);
            uint8_t * base = nullptr;
            ggml_backend_buffer_type_t buft_staging = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(devices[groups[0].id].backend));
            if (buft_staging != nullptr) {
                staging_buf.reset(ggml_backend_buft_alloc_buffer(buft_staging, staging_thread_size*n_io_threads));
                if (staging_buf) {
                    base = (uint8_t *) ggml_backend_buffer_get_base(staging_buf.get());
                } else {
                    LLAMA_LOG_WARN("llama_moe_cache: failed to allocate %s staging buffers, using plain memory\n", ggml_backend_buft_name(buft_staging));
                }
            }
            if (base == nullptr) {
                staging_mem.resize(staging_thread_size*n_io_threads);
                base = staging_mem.data();
            }
            for (int i = 0; i < n_io_threads; ++i) {
                staging.push_back(base + i*staging_thread_size);
            }

            open_server_files();
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %10s LongHaul cache size = %8.2f MiB for %.2f MiB of experts on disk, %d I/O threads, %d tokens reserved per group%s\n",
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0, io_pool ? io_pool->n_threads() : 0, d.n_ubatch_reserved,
                d.server_files.empty() ? "" : ", read by the server");
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("llama_moe_cache: %2zu layers (%d with own experts), %s: %5d slots (%.1f%%, %.1f tokens worth)\n",
                        g.layers.size(), g.n_owners, ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.n_owners*g.ref[0]->ne[2]),
                        (double) g.n_slots/(g.layers.size()*n_expert_used));
                }
            }
        }
    }

    // the RPC devices whose servers have the model files read the missing experts themselves, the others get them from this host
    void open_server_files() {
        for (device & d : devices) {
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            if (d.host_bytes == 0 || reg == nullptr || strcmp(ggml_backend_reg_name(reg), "RPC") != 0) {
                continue;
            }
            auto open_fn = (decltype(ggml_backend_rpc_dev_open_file) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_rpc_dev_open_file");
            auto read_fn = (decltype(ggml_backend_rpc_dev_set_tensors_from_file) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_rpc_dev_set_tensors_from_file");
            if (open_fn == nullptr || read_fn == nullptr) {
                continue;
            }

            std::vector<int32_t> ids;
            std::string missing;
            for (const group & g : groups) {
                if (g.id != (int32_t) (&d - devices.data())) {
                    continue;
                }
                for (int32_t il : g.layers) {
                    for (const ggml_tensor * w : layers[il].experts) {
                        size_t file = 0;
                        size_t offs = 0;
                        if (!model.longhaul_src(w, file, offs)) {
                            missing = w->name;
                            break;
                        }
                        if (file >= ids.size()) {
                            ids.resize(file + 1, -1);
                        }
                        if (ids[file] >= 0) {
                            continue;
                        }
                        const std::string & path = model.longhaul_path(file);
                        ids[file] = path.empty() ? -1 : open_fn(dev, path.c_str());
                        if (ids[file] < 0) {
                            missing = path.empty() ? "a model file opened without a path" : path;
                            break;
                        }
                    }
                    if (!missing.empty()) {
                        break;
                    }
                }
                if (!missing.empty()) {
                    break;
                }
            }
            if (!missing.empty()) {
                LLAMA_LOG_INFO("llama_moe_cache: %s has no copy of %s (see ggml-rpc-server --model-dir), the experts are sent from this host\n",
                    ggml_backend_dev_name(dev), missing.c_str());
                continue;
            }
            d.server_files = std::move(ids);
            rpc_set_from_file = read_fn;
        }
    }

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    int64_t max_ubatch() const {
        int64_t res = INT64_MAX;
        for (const group & g : groups) {
            if (g.n_slots < g.ref[0]->ne[2]) {
                res = std::min<int64_t>(res, g.n_slots/n_expert_used);
            }
        }
        return res;
    }

    // trace format, little endian: "MCT1", then u32 n_layer, n_expert_used, n_group
    // per group: u32 device, n_expert, n_layers, the layers (u32), f64 split, split_sum, u64 group host bytes, device host bytes, alignment,
    //            u32 n_tensors, the bytes of one expert of each tensor (u64)
    // then one record per slot lookup: i32 layer, experts per token, n_ids, the ids (i32)
    void write_trace_header(double split_sum) {
        auto u32 = [&](uint32_t v) { fwrite(&v, sizeof(v), 1, trace); };
        auto u64 = [&](uint64_t v) { fwrite(&v, sizeof(v), 1, trace); };
        auto f64 = [&](double   v) { fwrite(&v, sizeof(v), 1, trace); };
        fwrite("MCT1", 1, 4, trace);
        u32(layers.size());
        u32(n_expert_used);
        u32(groups.size());
        for (const group & g : groups) {
            const device & d = devices[g.id];
            u32(g.id);
            u32(g.ref[0]->ne[2]);
            u32(g.layers.size());
            for (int32_t il : g.layers) {
                u32(il);
            }
            f64(d.split);
            f64(split_sum);
            u64(g.host_bytes);
            u64(d.host_bytes);
            u64(ggml_backend_buft_get_alignment(d.buft));
            u32(g.ref.size());
            for (const ggml_tensor * t : g.ref) {
                u64(t->nb[2]);
            }
        }
    }

    void write_trace(int32_t il, int32_t n_used) {
        const int32_t rec[3] = { il, n_used, (int32_t) ids.size() };
        fwrite(rec, sizeof(int32_t), 3, trace);
        fwrite(ids.data(), sizeof(int32_t), ids.size(), trace);
    }

    ~impl() {
        log_stats();
        if (trace != nullptr) {
            fclose(trace);
        }
    }

    // the slot map of layer il for n_tokens tokens, nullptr if the layer is not cached
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0 || n_tokens == 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // the experts on disk have no other path, so the context keeps the ubatch within max_ubatch()
        if (std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            GGML_ABORT("LongHaul: a ubatch of %" PRId64 " tokens selects more experts in layer %d than the %d slots of the cache",
                n_tokens, il, groups[l.ig].n_slots);
        }
        return l.slot_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second : nullptr;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il = it->second;
        const layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, unless the lookup is the first node of its split
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));
        GGML_ASSERT(sel->op == GGML_OP_RESHAPE); // of the [n_expert_used, n_tokens] ids, see build_lookup

        const int32_t n_used = (int32_t) sel->src[0]->ne[0];

        // this also waits for the earlier splits, so the slots they read can be refilled
        ids.resize(ggml_nelements(sel));
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);

        if (trace != nullptr) {
            write_trace(il, n_used);
        }

        size_t n_hit = 0;
        if (!g.lru.plan(l.owner, ids.data(), ids.size(), sel->src[0]->ne[1], il == l.owner, fills, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits    += n_hit;
        st.misses  += fills.size();

        // the server reads the missing experts from its copies of the model files, in order with the graph computes
        const device & d = devices[g.id];
        if (!d.server_files.empty()) {
            server_reads.clear();
            size_t bytes = 0;
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * w = l.experts[ip];
                size_t file = 0;
                size_t offs = 0;
                GGML_ASSERT(model.longhaul_src(w, file, offs) && file < d.server_files.size() && d.server_files[file] >= 0);
                const size_t expert_size = w->nb[2];
                for (size_t i = 0; i < fills.size();) {
                    size_t n = 1;
                    while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                        n++;
                    }
                    server_reads.push_back({ g.banks[ip], fills[i].slot*expert_size, d.server_files[file], offs + fills[i].expert*expert_size, n*expert_size });
                    bytes += n*expert_size;
                    i += n;
                }
            }
            rpc_set_from_file(ggml_backend_get_device(backend), server_reads.data(), server_reads.size());
            st.bytes        += bytes;
            st.bytes_server += bytes;

            // the next copy synchronizes the backend before it changes the slot map again
            ggml_backend_tensor_set_async(backend, dst, g.lru.slot_map[l.owner], 0, ggml_nbytes(l.slot_map));
            return true;
        }

        // read the missing experts from the model files, consecutive experts going to consecutive slots are read together
        const int64_t t_start_us = ggml_time_us();
        size_t bytes = 0;
        jobs.clear();
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            const size_t max_run     = std::max<size_t>(1, staging_thread_size/expert_size);
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && n < max_run && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                jobs.push_back({ w, bank, fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size });
                bytes += n*expert_size;
                i += n;
            }
        }
        io_pool->run(jobs.size(), [&](size_t j, int thread) {
            const read_job & job = jobs[j];
            uint8_t * buf = staging[thread];
            model.longhaul_read(job.w, buf, job.offs_src, job.size);
            std::lock_guard<std::mutex> lock(set_mtx);
            ggml_backend_tensor_set(job.bank, buf, job.offs_dst, job.size);
        });

        st.bytes   += bytes;
        st.read_us += ggml_time_us() - t_start_us;

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, g.lru.slot_map[l.owner], 0, ggml_nbytes(l.slot_map));

        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            const double mib       = st.bytes/1024.0/1024.0;
            const double mib_local = (st.bytes - st.bytes_server)/1024.0/1024.0;
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, read = %.2f MiB, local %.2f MiB in %.2f s (%.1f MiB/s)\n",
                name, st.hits, st.misses, 100.0*st.hits/n, mib, mib_local, st.read_us/1e6, st.read_us > 0 ? mib_local/(st.read_us/1e6) : 0.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size) :
    pimpl(new impl(model, no_alloc, backends, bufts, size)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

int64_t llama_moe_cache::max_ubatch() const {
    return pimpl->max_ubatch();
}

llama_moe_cache_lookup llama_moe_cache::build_lookup(const llm_graph_context & g, const llama_moe_cache_layer & layer, ggml_tensor * selected_experts) const {
    const int32_t il = layer.il;
    ggml_context * ctx0 = g.ctx0;

    ggml_tensor * slot_map = pimpl->get_slot_map(il, selected_experts->ne[1], selected_experts->ne[0]);
    if (slot_map == nullptr) {
        return {};
    }
    // the experts on disk have no other path, so each check below is fatal
    for (ggml_tensor * w : { layer.up_exps, layer.gate_exps, layer.down_exps, layer.gate_up_exps }) {
        if (w == nullptr) {
            continue;
        }
        if (pimpl->get_experts(w) == nullptr) {
            GGML_ABORT("LongHaul: %s is not in the expert cache", w->name);
        }
    }

    // always a copy: the top-k view is contiguous for 1 token only, and the graph topology must not depend on the ubatch size
    ggml_tensor * ids = ggml_reshape_1d(ctx0, ggml_cont(ctx0, selected_experts), ggml_nelements(selected_experts));

    // the slot map is a host weight, so the scheduler starts a new split here and copies it with the copy callback
    // the callback reads the selected experts, reads the missing ones from disk and updates the slot map
    ggml_tensor * slots = ggml_get_rows(ctx0, slot_map, ids); // [1, n_expert_used*n_tokens]
    if (!ggml_backend_supports_op(pimpl->backend(il), slots)) {
        GGML_ABORT("LongHaul: the GPU does not support the slot lookup");
    }
    ggml_backend_sched_set_tensor_backend(g.sched, slots, pimpl->backend(il));
    g.cb(slots, "ffn_moe_slots", il);

    ggml_build_forward_expand(g.gf, slots);

    return { il, ggml_reshape_2d(ctx0, slots, selected_experts->ne[0], selected_experts->ne[1]) };
}

void llama_moe_cache::check_op(const ggml_tensor * w, const ggml_tensor * op) const {
    // the device of the cache must compute the matmul, else the scheduler copies the cache to another backend
    GGML_ASSERT(pimpl->get_experts(w) != nullptr);
    const auto & g = pimpl->groups[pimpl->group_of.at(w)];
    if (!ggml_backend_supports_op(pimpl->devices[g.id].backend, op)) {
        GGML_ABORT("LongHaul: the GPU does not support %s for %s (%s)", ggml_op_desc(op), w->name, ggml_type_name(w->type));
    }
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size + pimpl->staging_mem.size();
    }
    if (pimpl->staging_buf) {
        res[ggml_backend_buffer_get_type(pimpl->staging_buf.get())] += ggml_backend_buffer_get_size(pimpl->staging_buf.get());
    }
    return res;
}
