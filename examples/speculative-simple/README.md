# llama.cpp/examples/speculative-simple

Demonstration of basic greedy speculative decoding

See [docs/speculative.md](../../docs/speculative.md) for the supported draft types.

```bash
# spec-type draft-dspark (draft-dflash is an alias)
# -hf also downloads the dspark-*.gguf draft of the repo
./bin/llama-speculative-simple \
        -hf AtomicChat/Inkling-2-Mini-GGUF \
        -p "Here is a quick sort implementation in C++. Just code, no comments:\n\n#include" \
        --spec-type draft-dspark --spec-draft-n-max 7 -ngld 99 --color on \
        -n 256 --temp 0 --top-k 1 --seed 42 -ngl 99 -lv 4

# spec-type draft-dspark with local files
./bin/llama-speculative-simple \
        -m Inkling-2-Mini-AD-Q4_K.gguf \
        -md dspark-Inkling-2-Mini-Q8_0.gguf \
        -p "Here is a quick sort implementation in C++. Just code, no comments:\n\n#include" \
        --spec-type draft-dspark --spec-draft-n-max 7 -ngld 99 --color on \
        -n 256 --temp 0 --top-k 1 --seed 42 -ngl 99 -lv 4

# spec-type ngram-simple (no draft model)
./bin/llama-speculative-simple \
        -hf AtomicChat/Inkling-2-Mini-GGUF \
        -p "Here is a quick sort implementation in C++. Just code, no comments:\n\n#include" \
        --spec-type ngram-simple --color on \
        -n 256 --temp 0 --top-k 1 --seed 42 -ngl 99 -lv 4
```
