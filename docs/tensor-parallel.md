# Tensor parallelism over several Macs

With `-sm tensor` two Macs run one model as if it were on one machine: every large matrix of every layer is split between the two GPUs, so the memory bandwidth and the GPU compute of both Macs add up on every token. The price is a sync between the Macs inside every layer. Whether that pays off depends on the cable, so measure it first.

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
