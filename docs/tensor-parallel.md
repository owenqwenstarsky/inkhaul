# Tensor parallelism over several Macs

With `-sm tensor` two Macs run one Qwen 3.5 model (`qwen35`, `qwen35moe`) or Inkling 2 model (`inkling2`, `inkling2_loop`) as if it were on one machine: every large matrix of every layer is split between the two GPUs, so the memory bandwidth and the GPU compute of both Macs add up on every token. The price is a sync between the Macs inside every layer. Whether that pays off depends on the cable, so measure it first.

See [TENSOR_PARALLEL_PLAN.md](../TENSOR_PARALLEL_PLAN.md) for the design.

## Measure first: `llama-tp-probe`

Time per generated token with the work split in half is about `T1/2 + N*t`: `T1` the time on one Mac that holds the model, `N` the syncs per token (one or two per layer pass), `t` the cost of one sync. `llama-tp-probe` measures `t` and estimates the rest from your model file.

```sh
# 1. the link: on the remote Mac
llama-tp-probe listen -H 10.0.0.2 -p 50100
#    and on the main Mac (round trip and same-time exchange at 8 KiB, 64 KiB, 2 MiB, blocking and spinning receive)
llama-tp-probe connect -H 10.0.0.2 -p 50100

# 2. the GPU: the time to run and wait for a tiny graph on this Mac, and the GPU part of one async sync
llama-tp-probe gpu
#    the same with each graph reading 64 MiB of weights, closer to a real layer
llama-tp-probe gpu --work 2

# 3. the model: bytes read per token by part, and the estimated speed of each mode for a few sync costs
llama-tp-probe budget -m ~/models/Inkling-2-Mini-AD-Q4_K.gguf --bw 120 --t 50,100,300
```

- Bind `listen` to the Thunderbolt address, never to `0.0.0.0`. It answers one `connect` and exits.
- `connect` prints the round trip, the one-way latency (half of it) and the same-time exchange per round, where both Macs send and then receive, which is what a sync does.
- `gpu` prints:
  - the time of a graph run plus the wait for it, which the blocking all-reduce pays at every sync;
  - the busy-queue turnaround: the host reaches an event that the GPU waits for, the GPU runs the graph and reaches an event that the host spins on. Minus the GPU time of the graph, this is what the async all-reduce costs the GPU per sync, before the link. It is an upper bound, because back-to-back graphs can overlap on the GPU; with `--work 0` it is close.
- `budget` reads the GGUF header only. `--bw` is the memory bandwidth of one Mac in GB/s, `--t` the sync costs to compare in microseconds. The last column is the sync cost above which a mode is slower than one Mac.
- `llama-tp-probe self-test` checks the tool itself over `127.0.0.1`.

The sustained transfers of the 2 MiB test push a lot of data over the cable. One published test saw Macs become unstable under heavy TCP load over Thunderbolt, so start with `-s 8K` and a small `-n`.

## Running `-sm tensor` on two Macs

The remote Mac runs a `ggml-rpc-server`. The main Mac runs `llama-cli` or `llama-server`, which with `--tp-host` also runs the server of its own GPU (rank 0) on a thread. The two servers add up their partial results directly between each other, and only the input and the logits pass through the client. The cable setup is the one in [tools/rpc](../tools/rpc/README.md#this-fork-several-macs-over-thunderbolt).

```sh
# on the remote Mac
ggml-rpc-server -H 10.0.0.2 -p 50052

# on the main Mac: rank 0 on its Thunderbolt address, the remote Mac as rank 1
llama-cli -m ~/models/Qwen3.5-35B-A3B-Q4_K_M.gguf --rpc 10.0.0.2:50052 -sm tensor --tp-host 10.0.0.1 --tp-attn mirror

# the same with a second process for rank 0 instead of --tp-host
ggml-rpc-server -H 10.0.0.1 -p 50052 &
llama-cli -m ~/models/Qwen3.5-35B-A3B-Q4_K_M.gguf --rpc 10.0.0.1:50052,10.0.0.2:50052 -dev RPC0,RPC1 -sm tensor --tp-attn mirror
```

- Rank 0 listens for the other server on its port + 1000 (here `10.0.0.1:51052`), on the address it was started with. Allow that port in the macOS firewall, or pick another one with `GGML_RPC_COMM_PORT` on the client.
- `--tp-host ADDR[:PORT]` (default port 50052) serves the first local GPU. The address must be the one the remote Mac reaches this Mac on, not `127.0.0.1`. The devices become that server and then the `--rpc` servers, so do not give `-dev`.
- Without `--tp-host`, give only RPC devices with `-dev`. With the local GPU in the list the partial results go through the client, which is slow, and the log warns.
- Each large matrix is split by `-ts` (default: evenly). The experts, the shared experts and the dense FFN are split by rows, in steps of 128 rows (256 for K-quants). The router and the norms are on both Macs. The load log prints the share that each class of tensors got after the rounding, for example `tensor split: experts 50/50 (n_ff 512, step 256), shared experts 50/50 (n_ff 512, step 256), attention mirrored, output 50/50`.
- `--tp-attn` sets what happens to attention (and to the Gated DeltaNet layers of Qwen 3.5):
  - `split` (default): heads, KV cache and recurrent state are split by groups of heads. Two syncs per layer. Attention reads and the KV cache are halved, so the possible context doubles.
  - `mirror`: both Macs hold all attention weights and the whole KV cache and compute attention in full. One sync per layer, the one after the experts. Attention is not faster and its memory is doubled.
  - Which one is faster depends on the cost of a sync, see `llama-tp-probe budget`. Above about 100 us per sync `mirror` should win, and between two Macs over Thunderbolt a sync costs more than that. `llama-bench -tpa split,mirror` compares both.
- The shared experts add no sync: the meta backend adds their partial to the one of the routed experts and syncs both at once.
- All experts stay in memory, split between the Macs: LongHaul (experts on disk) is off, and `-lhc`, `-lhcd` and `-es` do not apply.
- Flash attention is turned on (it cannot be off). Backend sampling is off. `--fit` skips this mode with a warning, so set `-c` yourself.
- The model must fit in the memory of both Macs together, with each Mac holding its share plus the mirrored tensors.
- Only `qwen35`, `qwen35moe`, `inkling2` and `inkling2_loop` support this mode for now. Other models fail to load with `-sm tensor`.
- Inkling 2 always mirrors attention, as with `--tp-attn mirror`, and warns if `split` was given: each layer keeps the state of its K/V convolutions and of its post convolutions in one tensor, and the meta backend cannot split one part of a tensor and copy the other. Its experts are split, so it syncs once per layer pass (48 per token for Inkling 2 Mini). Its output matrix is on both Macs too, because the model pads the vocabulary and cuts the logits back, which a split of the vocabulary cannot follow.

### How the servers sync

On Metal the sync does not stop the GPU queue:
- After each split part of a layer, the GPU copies its partial result to a host buffer and marks an event.
- A thread of the server sends the partial to the other server and receives the other partial at the same time.
- The GPU waits for that thread, then adds the two partials.
- Meanwhile the server thread has already queued the next part, so the GPU starts it as soon as the sum is done.

Payloads of 32768 values or more (prompt processing) go over the wire in bf16. Each server rounds its own partial to bf16 too, so both Macs get the same bits.

| Variable | Default | Meaning |
|---|---|---|
| `GGML_RPC_COMM_TIMEOUT_MS` | `30000` | A server whose peer does not answer for this long stops the run with an error, instead of hanging (TCP only). `0` waits forever. |
| `GGML_RPC_COMM_SPIN_US` | `500` | How long the sync thread spins before it sleeps, while it waits for the GPU and for small messages. Spinning costs a CPU core, sleeping costs tens of microseconds per wake-up. |
| `GGML_RPC_COMM_SYNC` | unset | Set it to use the blocking sync: the server waits for its GPU, exchanges, then queues the add. Use it to compare, or for backends without host events. |
| `GGML_RPC_COMM_PORT` | port of rank 0 + 1000 | Set on the client: the port rank 0 listens on for the other servers. |

- Both servers need protocol 10.3 or newer (this build). With an older server the client logs a warning and the partials go through the client.
- Each Mac runs the router of every MoE layer itself. On two Macs with the same chip both pick the same experts, bit for bit. To be sure, each server sends a hash of the expert ids of the layer with its partial (8 more bytes, no extra trip), and a mismatch stops the run with an error that names the layer, for example `the ranks picked different experts in ffn_moe_topk-12`. On two different chips the GPUs can round the router differently; that case is not supported yet.
- Two servers on one Mac must use different devices (for example `GGML_METAL_DEVICES=2` and `-d MTL0`, `-d MTL1`). On one device both servers share one GPU queue, so each GPU would wait for the other forever; Metal then stops the queue with a GPU timeout error.
- Over RDMA (Thunderbolt 5) the servers keep the ordered exchange (rank 0 sends first); the same-time exchange is for TCP.
- The client sends the small commands of a pass in fewer TCP sends: each command goes out in one send, and the commands already queued behind it share that send.

## Speculative decoding

A pass checks all the guessed tokens of a draft at once, so it pays the same syncs per layer for several tokens. That makes the split pass cheaper per token, and it is the best way to get more tokens per second out of two Macs on one stream.

- n-gram speculation (`--spec-type ngram-simple`, `ngram-mod`, ...) needs no extra model and no extra memory. It helps when the output repeats the prompt (code edits, summaries with quotes).
- A draft model (`-md`) runs whole on the first local GPU by default (the GPU behind `--tp-host`), not split and not through RPC: a small model cannot earn back a sync per layer. Set `-devd` to put it elsewhere.
- DSpark (`--spec-type draft-dspark`) reads the inputs of some target layers. With a split target those come from host copies, because the draft is on another device than the target's meta buffer. This works for Qwen 3.5 targets (tested). Inkling 2 targets do not record layer inputs in this build, so DSpark stops with an assert there, also on one Mac.
- The gain on two Macs is not measured yet. Compare with `llama-bench` or `llama-speculative` before and after `--spec-type`.
