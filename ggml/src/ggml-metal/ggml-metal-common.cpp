#include "ggml-metal-common.h"
#include "ggml-metal-fusion.h"

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

// must stay in sync with the kernel_fwht_<type>_<N> templates in misc.metal. Widths up to
// 512 run on the simdgroup kernel and need no threadgroup memory. The wider ones allocate
// float[N] per threadgroup, so they are only available where that fits.
static bool ggml_metal_fwht_supported_size(int64_t n, size_t max_tg_mem) {
    if (n == 64 || n == 128 || n == 256 || n == 512) {
        return true;
    }

    if (n == 1024 || n == 2048 || n == 4096 || n == 8192) {
        return (size_t) n * sizeof(float) <= max_tg_mem;
    }

    return false;
}

// the FWHT kernels handle a Hadamard-hinted MUL_MAT only under these conditions. supports_op
// and the dispatch must ask the same question: an F16 src1 that is admitted but then falls
// through reaches the generic path, which has no F32 src0 by F16 src1 kernel.
bool ggml_metal_op_mul_mat_use_fwht(const struct ggml_tensor * op, size_t max_tg_mem) {
    return ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD && op->type == GGML_TYPE_F32 &&
           (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && ggml_is_contiguous(op->src[1]) &&
           ggml_is_contiguous(op) && ggml_are_same_shape(op->src[1], op) &&
           ggml_metal_fwht_supported_size(op->src[1]->ne[0], max_tg_mem);
}

bool ggml_metal_op_mul_mat_use_mm(const struct ggml_tensor * op, bool has_simdgroup_mm) {
    const int64_t ne00 = op->src[0]->ne[0];
    const int64_t ne11 = op->src[1]->ne[1];

    return !ggml_is_transposed(op->src[0]) &&
           !ggml_is_transposed(op->src[1]) &&
           has_simdgroup_mm && ne11 > 8 && ne00 >= 32;
}

bool ggml_metal_op_mul_mat_id_use_mm(const struct ggml_tensor * op, bool has_simdgroup_mm) {
    const int64_t ne00 = op->src[0]->ne[0];
    const int64_t ne02 = op->src[0]->ne[2];
    const int64_t ne21 = op->src[2]->ne[1];

    return has_simdgroup_mm && ne00 >= 64 && ne02 <= 1024 && ne21 >= 32;
}

// the most src1 rows of the few-row MMA kernels
static constexpr int64_t GGML_METAL_MMA_ROWS_MAX = 16;

// src1 rows per 8x8 simdgroup matrix tile of the few-row MMA kernels
static constexpr int64_t GGML_METAL_MMA_TILE_ROWS = 8;

// weights per K step of the q5_K and generic few-row MMA kernels
static constexpr int64_t GGML_METAL_MMA_K_CHUNK = 64;

enum ggml_metal_mma_kind ggml_metal_mul_mv_mma_kind(enum ggml_type type, int rt) {
    if (type == GGML_TYPE_Q4_0 || (type == GGML_TYPE_Q8_0 && rt == 1)) {
        return GGML_METAL_MMA_KIND_BLK;
    }
    return type == GGML_TYPE_Q5_K ? GGML_METAL_MMA_KIND_Q5_K : GGML_METAL_MMA_KIND_GEN;
}

int ggml_metal_mul_mv_mma_rt(const struct ggml_tensor * op) {
    return op->src[1]->ne[1] > GGML_METAL_MMA_TILE_ROWS ? 2 : 1;
}

static bool ggml_metal_mul_mv_mma_type_supported(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_TQ2_0:
            return true;
        default:
            return false;
    }
}

int64_t ggml_metal_mul_mv_mma_k_step(enum ggml_type type, int rt) {
    if (!ggml_metal_mul_mv_mma_type_supported(type)) {
        return 0;
    }
    return ggml_metal_mul_mv_mma_kind(type, rt) == GGML_METAL_MMA_KIND_BLK ? ggml_blck_size(type) : GGML_METAL_MMA_K_CHUNK;
}

static bool ggml_metal_mul_mat_mma_type_ok(const struct ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const int64_t step = ggml_metal_mul_mv_mma_k_step(src0->type, ggml_metal_mul_mv_mma_rt(op));

    return step > 0 && src0->ne[0] % step == 0 && src0->nb[0] == ggml_type_size(src0->type);
}

// the fewest src1 rows at which the few-row MMA kernels beat the mat-vec kernels (measured on an M3 Ultra)
static int64_t ggml_metal_mul_mv_mma_rows_min(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return 6;
        case GGML_TYPE_TQ2_0:
            return 5;
        case GGML_TYPE_BF16:
            return 4;
        case GGML_TYPE_F16:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            return 3;
        default:
            return 2;
    }
}

bool ggml_metal_op_mul_mat_use_mma(const struct ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    // the batch shape goes into int16 function constants
    const bool batch_ok = src1->ne[2] <= INT16_MAX && src1->ne[2]/src0->ne[2] <= INT16_MAX && src1->ne[3]/src0->ne[3] <= INT16_MAX;

    return ggml_metal_mul_mat_mma_type_ok(op) && batch_ok &&
        src1->type == GGML_TYPE_F32 && src1->ne[1] >= ggml_metal_mul_mv_mma_rows_min(src0->type) && src1->ne[1] <= GGML_METAL_MMA_ROWS_MAX &&
        !ggml_is_transposed(src0) && !ggml_is_transposed(src1) &&
        src1->nb[0] == sizeof(float) && src1->nb[1] % 16 == 0 && src1->nb[2] % 16 == 0 && src1->nb[3] % 16 == 0;
}

bool ggml_metal_op_mul_mat_may_use_mma(const struct ggml_tensor * op) {
    return ggml_metal_mul_mv_mma_type_supported(op->src[0]->type) && op->src[1]->type == GGML_TYPE_F32;
}

// represents a memory range (i.e. an interval from a starting address p0 to an ending address p1 in a given buffer pb)
// the type indicates whether it is a source range (i.e. ops read data from it) or a destination range (i.e. ops write data to it)
struct ggml_mem_range {
    uint64_t pb; // buffer id

    uint64_t p0; // begin
    uint64_t p1; // end

    ggml_mem_range_type pt;
};

struct ggml_mem_ranges {
    std::vector<ggml_mem_range> ranges;

    int debug = 0;
};

ggml_mem_ranges_t ggml_mem_ranges_init(int debug) {
    auto * res = new ggml_mem_ranges;

    res->ranges.reserve(256);
    res->debug = debug;

    return res;
}

void ggml_mem_ranges_free(ggml_mem_ranges_t mrs) {
    delete mrs;
}

void ggml_mem_ranges_reset(ggml_mem_ranges_t mrs) {
    mrs->ranges.clear();
}

static bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    mrs->ranges.push_back(mr);

    return true;
}

static ggml_mem_range ggml_mem_range_from_tensor(const ggml_tensor * tensor, ggml_mem_range_type pt) {
    // always use the base tensor
    tensor = tensor->view_src ? tensor->view_src : tensor;

    GGML_ASSERT(!tensor->view_src);

    ggml_mem_range mr;

    if (tensor->buffer) {
        // when the tensor is allocated, use the actual memory address range in the buffer
        //
        // take the actual allocated size with ggml_backend_buft_get_alloc_size()
        // this can be larger than the tensor size if the buffer type allocates extra memory
        // ref: https://github.com/ggml-org/llama.cpp/pull/15966
        mr = {
            /*.pb =*/ (uint64_t) tensor->buffer,
            /*.p0 =*/ (uint64_t) tensor->data,
            /*.p1 =*/ (uint64_t) tensor->data + ggml_backend_buft_get_alloc_size(tensor->buffer->buft, tensor),
            /*.pt =*/ pt,
        };
    } else {
        // otherwise, the pointer address is used as an unique id of the memory ranges
        //   that the tensor will be using when it is allocated
        mr = {
            /*.pb =*/ (uint64_t) tensor,
            /*.p0 =*/ 0,    //
            /*.p1 =*/ 1024, // [0, 1024) is a dummy range, not used
            /*.pt =*/ pt,
        };
    };

    return mr;
}

static ggml_mem_range ggml_mem_range_from_tensor_src(const ggml_tensor * tensor) {
    return ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_SRC);
}

static ggml_mem_range ggml_mem_range_from_tensor_dst(const ggml_tensor * tensor) {
    return ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_DST);
}

static bool ggml_mem_ranges_add_src(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    if (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add src range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    return ggml_mem_ranges_add(mrs, mr);
}

static bool ggml_mem_ranges_add_dst(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    if (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add dst range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    return ggml_mem_ranges_add(mrs, mr);
}

bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i]) {
            ggml_mem_ranges_add_src(mrs, tensor->src[i]);
        }
    }

    return ggml_mem_ranges_add_dst(mrs, tensor);
}

static bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    for (size_t i = 0; i < mrs->ranges.size(); i++) {
        const auto & cmp = mrs->ranges[i];

        // two memory ranges cannot intersect if they are in different buffers
        if (mr.pb != cmp.pb) {
            continue;
        }

        // intersecting source ranges are allowed
        if (mr.pt == MEM_RANGE_TYPE_SRC && cmp.pt == MEM_RANGE_TYPE_SRC) {
            continue;
        }

        if (mr.p0 < cmp.p1 && mr.p1 > cmp.p0) {
            if (mrs->debug > 2) {
                GGML_LOG_DEBUG("%s: the %s range buf=%lld, [%lld, %lld) overlaps with a previous %s range buf=%lld, [%lld, %lld)\n",
                        __func__,
                        mr.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        mr.pb, mr.p0, mr.p1,
                        cmp.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        cmp.pb, cmp.p0, cmp.p1);
            }

            return false;
        }
    }

    return true;
}

static bool ggml_mem_ranges_check_src(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    const bool res = ggml_mem_ranges_check(mrs, mr);

    return res;
}

static bool ggml_mem_ranges_check_dst(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    const bool res = ggml_mem_ranges_check(mrs, mr);

    return res;
}

bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i]) {
            if (!ggml_mem_ranges_check_src(mrs, tensor->src[i])) {
                return false;
            }
        }
    }

    return ggml_mem_ranges_check_dst(mrs, tensor);
}

struct node_info {
    ggml_tensor * node;

    std::vector<ggml_tensor *> fused;

    ggml_op op() const {
        return node->op;
    }

    ggml_tensor * dst() const {
        return fused.empty() ? node : fused.back();
    }

    bool is_empty() const {
        return ggml_op_is_empty(node->op);
    }

    void add_fused(ggml_tensor * t) {
        fused.push_back(t);
    }

    bool is_output(const ggml_tensor * t) const {
        if (t == node) {
            return true;
        }
        for (const auto * f : fused) {
            if (t == f) {
                return true;
            }
        }
        return false;
    }
};

// the ops that the level reorder can move: their memory accesses are their sources and their destination
// (with the extra memory of the destination, see ggml_backend_metal_buffer_type_get_alloc_size)
static bool ggml_metal_graph_optimize_movable_op(ggml_op op) {
    switch (op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_L2_NORM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN:
        case GGML_OP_CLAMP:
        case GGML_OP_TRI:
        case GGML_OP_DIAG:
        case GGML_OP_MUL:
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_DIV:
        case GGML_OP_GLU:
        case GGML_OP_SCALE:
        case GGML_OP_UNARY:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_ARGSORT:
        case GGML_OP_GET_ROWS:
        case GGML_OP_SET_ROWS:
        case GGML_OP_SET:
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_REPEAT:
        case GGML_OP_CONCAT:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_PAD:
        case GGML_OP_FILL:
        case GGML_OP_LOG:
        case GGML_OP_FLASH_ATTN_EXT:
            return true;
        default:
            return ggml_op_is_empty(op);
    }
}

// a node and the nodes fused with it can move if all their ops can
static bool ggml_metal_graph_optimize_movable(const node_info & node) {
    if (!ggml_metal_graph_optimize_movable_op(node.op())) {
        return false;
    }
    for (const auto * f : node.fused) {
        if (!ggml_metal_graph_optimize_movable_op(f->op)) {
            return false;
        }
    }
    return true;
}

// order the nodes by level: a node goes one level after the last node that it must follow, so that the nodes of a level
// can run together and the encoder puts one barrier between two levels
// a node must follow an earlier node if it reads what the earlier node writes, or writes what the earlier node reads or writes
// each node takes the latest level that its followers allow, so that its result is made shortly before its first use
// a node with an op that cannot move splits the graph: the nodes before it stay before it
// level gets the level of each node, a segment after an op that cannot move starts at a higher level than the ones before it
static std::vector<int> ggml_metal_graph_optimize_levels(const std::vector<node_info> & nodes, std::vector<int> & level) {
    const int n = nodes.size();

    std::vector<int> res;
    res.reserve(n);

    // the earlier nodes that a node must follow
    std::vector<std::vector<int>> preds(n);
    level.assign(n, 0);

    // the first level of the current segment
    int base = 0;

    // the accesses of the current segment, per buffer, or per tensor if the tensor is not allocated
    struct access {
        uint64_t p0;
        uint64_t p1;
        int      node;
    };
    struct accesses {
        std::vector<access> writes;
        std::vector<access> reads;
    };
    std::unordered_map<uint64_t, accesses> acc;

    // the node that makes a tensor, for the views
    std::unordered_map<const ggml_tensor *, int> maker;

    const auto add_pred = [&](int i, int p) {
        if (p != i && std::find(preds[i].begin(), preds[i].end(), p) == preds[i].end()) {
            preds[i].push_back(p);
        }
    };

    const auto schedule = [&](int i0, int i1) {
        acc.clear();
        maker.clear();

        int n_level = 0;

        for (int i = i0; i < i1; i++) {
            const node_info & node = nodes[i];

            maker[node.node] = i;
            for (const auto * f : node.fused) {
                maker[f] = i;
            }

            if (node.is_empty()) {
                continue;
            }

            // the reads: the sources from outside of the fused group
            std::vector<ggml_mem_range> rd;
            const auto add_srcs = [&](const ggml_tensor * t) {
                for (int s = 0; s < GGML_MAX_SRC; s++) {
                    const ggml_tensor * src = t->src[s];
                    if (src && !node.is_output(src)) {
                        rd.push_back(ggml_mem_range_from_tensor_src(src));
                    }
                }
            };
            add_srcs(node.node);
            for (const auto * f : node.fused) {
                add_srcs(f);
            }

            // the writes: all the tensors of the fused group
            std::vector<ggml_mem_range> wr;
            wr.push_back(ggml_mem_range_from_tensor_dst(node.node));
            for (const auto * f : node.fused) {
                wr.push_back(ggml_mem_range_from_tensor_dst(f));
            }

            for (const auto & mr : rd) {
                auto it = acc.find(mr.pb);
                if (it == acc.end()) {
                    continue;
                }
                for (const auto & a : it->second.writes) {
                    if (mr.p0 < a.p1 && a.p0 < mr.p1) {
                        add_pred(i, a.node);
                    }
                }
            }
            for (const auto & mr : wr) {
                auto it = acc.find(mr.pb);
                if (it == acc.end()) {
                    continue;
                }
                for (const auto & a : it->second.writes) {
                    if (mr.p0 < a.p1 && a.p0 < mr.p1) {
                        add_pred(i, a.node);
                    }
                }
                for (const auto & a : it->second.reads) {
                    if (mr.p0 < a.p1 && a.p0 < mr.p1) {
                        add_pred(i, a.node);
                    }
                }
            }

            for (const auto & mr : rd) {
                acc[mr.pb].reads.push_back({ mr.p0, mr.p1, i });
            }
            for (const auto & mr : wr) {
                acc[mr.pb].writes.push_back({ mr.p0, mr.p1, i });
            }

            int l = 0;
            for (const int p : preds[i]) {
                l = std::max(l, level[p] + 1);
            }
            level[i] = l;
            n_level = std::max(n_level, l + 1);
        }

        // move each node to the latest level before its first follower
        for (int i = i0; i < i1; i++) {
            if (!nodes[i].is_empty()) {
                level[i] = n_level - 1;
            }
        }
        for (int i = i1 - 1; i >= i0; i--) {
            for (const int p : preds[i]) {
                level[p] = std::min(level[p], level[i] - 1);
            }
        }

        // a view goes with the node that makes its source, before the nodes that use it
        for (int i = i0; i < i1; i++) {
            const node_info & node = nodes[i];
            if (!node.is_empty()) {
                continue;
            }
            int l = -1;
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                const ggml_tensor * src = node.node->src[s];
                if (!src) {
                    continue;
                }
                auto it = maker.find(src);
                if (it != maker.end() && it->second < i) {
                    l = std::max(l, level[it->second]);
                }
            }
            level[i] = l;
        }

        std::vector<int> order;
        order.reserve(i1 - i0);
        for (int i = i0; i < i1; i++) {
            order.push_back(i);
        }
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return level[a] < level[b];
        });

        res.insert(res.end(), order.begin(), order.end());

        for (int i = i0; i < i1; i++) {
            level[i] += base + 1;
        }
        base += n_level + 1;
    };

    int i0 = 0;
    for (int i = 0; i < n; i++) {
        if (!ggml_metal_graph_optimize_movable(nodes[i])) {
            schedule(i0, i);
            res.push_back(i);
            level[i] = base++;
            i0 = i + 1;
        }
    }
    schedule(i0, n);

    GGML_ASSERT((int) res.size() == n);

    return res;
}

void ggml_graph_optimize(ggml_cgraph * gf, void (*add_alloc_dep)(void *, ggml_tensor *, ggml_tensor *), void * user_data) {
    const int n = gf->n_nodes;

    std::vector<node_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    //
    // the fusable sequences are declared in the fusion table (ggml-metal-fuse.cpp), so the
    // packing here is driven by the same patterns that the op encoders will later use
    for (int i = 0; i < n; i++) {
        node_info node = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        const int f = ggml_metal_fusion_max(gf, i);

        // add the fused tensors into the node info so we can unfuse them later
        for (int k = 1; k < f; k++) {
            ++i;

            // the .dst() becomes the last fused tensor
            node.add_fused(gf->nodes[i]);
        }

        nodes.push_back(std::move(node));
    }

#if 1
    // reorder to improve concurrency
    std::vector<int> level;
    const auto order = ggml_metal_graph_optimize_levels(nodes, level);

    // the nodes of a level run together: keep the sources of a level allocated until its last node,
    // else the allocator gives their memory to a later node of the same level and the encoder must put a barrier
    if (add_alloc_dep != nullptr) {
        // the last node of each level in the new order
        std::unordered_map<int, int> last;
        for (const int i : order) {
            if (!nodes[i].is_empty()) {
                last[level[i]] = i;
            }
        }

        // the last level that uses each tensor of the compute buffer, and the last node of that level that uses it
        std::unordered_map<ggml_tensor *, std::pair<int, int>> use;
        std::vector<ggml_tensor *> used; // in the order of the first use
        const auto keep = [&](ggml_tensor * t, int i) {
            t = t->view_src ? t->view_src : t;
            if (t->buffer) {
                return;
            }
            auto it = use.find(t);
            if (it == use.end()) {
                use[t] = { level[i], i };
                used.push_back(t);
            } else if (it->second.first <= level[i]) {
                it->second = { level[i], i };
            }
        };

        for (const int i : order) {
            const node_info & node = nodes[i];
            if (node.is_empty()) {
                continue;
            }

            // the sources from outside of the fused group
            const auto add_srcs = [&](const ggml_tensor * t) {
                for (int s = 0; s < GGML_MAX_SRC; s++) {
                    if (t->src[s] && !node.is_output(t->src[s])) {
                        keep(t->src[s], i);
                    }
                }
            };
            add_srcs(node.node);
            for (const auto * f : node.fused) {
                add_srcs(f);
            }

            // the encoder adds the dst of each fused node to its ranges, also when the fused kernel does not write it:
            // keep the dst of the nodes before the last one of the group allocated until the last node of the level too
            if (!node.fused.empty()) {
                if (!ggml_op_is_empty(node.node->op)) {
                    keep(node.node, i);
                }
                for (size_t k = 0; k + 1 < node.fused.size(); ++k) {
                    if (!ggml_op_is_empty(node.fused[k]->op)) {
                        keep(node.fused[k], i);
                    }
                }
            }
        }

        for (ggml_tensor * t : used) {
            const int l = use[t].first;
            const int i = use[t].second;
            if (last[l] != i) {
                add_alloc_dep(user_data, t, nodes[last[l]].dst());
            }
        }
    }
#else
    std::vector<int> order(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++) {
        order[i] = i;
    }
#endif

    // unfuse
    {
        int j = 0;
        for (const auto i : order) {
            const auto & node = nodes[i];

            gf->nodes[j++] = node.node;

            for (auto * fused : node.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
}
