# Tensor parallelism over several Macs

With `-sm tensor` two Macs run one Qwen 3.5 model (`qwen35`, `qwen35moe`) as if it were on one machine: every large matrix of every layer is split between the two GPUs, so the memory bandwidth and the GPU compute of both Macs add up on every token. The price is a sync between the Macs inside every layer. Whether that pays off depends on the cable, so measure it first.

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

Each Mac runs a `ggml-rpc-server`, and the main Mac also runs `llama-cli` or `llama-server` with only the two RPC devices. Then the two servers add up their partial results directly between each other, and only the input and the logits pass through the client. The cable setup is the one in [tools/rpc](../tools/rpc/README.md#this-fork-several-macs-over-thunderbolt).

```sh
# on the remote Mac
ggml-rpc-server -H 10.0.0.2 -p 50052

# on the main Mac: its own server, then the client with both servers
ggml-rpc-server -H 10.0.0.1 -p 50052 &
llama-cli -m ~/models/Qwen3.5-35B-A3B-Q4_K_M.gguf --rpc 10.0.0.1:50052,10.0.0.2:50052 -dev RPC0,RPC1 -sm tensor
```

- The first server in `--rpc` is rank 0. It listens for the other server on its port + 1000 (here `10.0.0.1:51052`), on the address it was started with. Allow that port in the macOS firewall.
- Give only RPC devices with `-dev`. With the local GPU in the list the partial results go through the client, which is slow, and the log warns.
- Each large matrix is split by `-ts` (default: evenly). Attention is split by groups of heads, the experts and the dense FFN by rows. The router, the norms and the shared experts are on both Macs.
- All experts stay in memory, split between the Macs: LongHaul (experts on disk) is off, and `-lhc`, `-lhcd` and `-es` do not apply.
- Flash attention is turned on (it cannot be off). Backend sampling is off. `--fit` skips this mode with a warning, so set `-c` yourself.
- The model must fit in the memory of both Macs together, with each Mac holding its share plus the mirrored tensors.
- Only `qwen35` and `qwen35moe` support this mode for now. Other models fail to load with `-sm tensor`.

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

- Both servers need protocol 10.2 or newer (this build). With an older server the client logs a warning and the partials go through the client.
- Two servers on one Mac must use different devices (for example `GGML_METAL_DEVICES=2` and `-d MTL0`, `-d MTL1`). On one device both servers share one GPU queue, so each GPU would wait for the other forever; Metal then stops the queue with a GPU timeout error.
- Over RDMA (Thunderbolt 5) the servers keep the ordered exchange (rank 0 sends first); the same-time exchange is for TCP.
- The client sends the small commands of a pass in fewer TCP sends: each command goes out in one send, and the commands already queued behind it share that send.
