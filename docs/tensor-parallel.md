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

# 2. the GPU: the time to run and wait for a tiny graph on this Mac
llama-tp-probe gpu

# 3. the model: bytes read per token by part, and the estimated speed of each mode for a few sync costs
llama-tp-probe budget -m ~/models/Inkling-2-Mini-AD-Q4_K.gguf --bw 120 --t 50,100,300
```

- Bind `listen` to the Thunderbolt address, never to `0.0.0.0`. It answers one `connect` and exits.
- `connect` prints the round trip, the one-way latency (half of it) and the same-time exchange per round, where both Macs send and then receive, which is what a sync does.
- `gpu` prints the time of a tiny graph run plus the wait for it. The blocking all-reduce pays it at every sync.
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

The sync is still the simple one: each server waits for its GPU, then exchanges the partial result with the other server. A faster sync is the next step of the plan.
