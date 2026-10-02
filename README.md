# go-tiny-llm

A character-level transformer language model written from scratch in Go, with no
frameworks, no dependencies for the core, small enough to read in an afternoon.
Companion code for the "Build a Tiny LLM in Go" article series.

Everything is written for clarity, not speed. If you can read Go, you can read
the whole model.

## Run it

```bash
go run ./cmd/stage1_bigram    # counting: predict the next letter (article 1)
go run ./cmd/stage2_context   # why counting longer context explodes (article 2)
go run ./cmd/stage3_mlp       # the first model that learns (article 3)
go run ./cmd/stage4_attention # attention weights, visualised (article 4)
go run ./cmd/stage5_train     # the full transformer trains + dreams (article 5)
go run ./cmd/stage6_distill   # a small student learns from a big teacher (article 6)
```

## Tune it

Every knob lives in `config.go`: context length, embedding width, number of
layers and heads, learning rate, training steps. Change one line, run again.

## Get your own data

`data/input.txt` ships ready to use. To train on something else:

```bash
go run ./cmd/scraper https://some-url-with-lots-of-text
```

## Tests

```bash
go test ./...
```

The autograd engine is checked against numerical gradients, so you can trust the
math even though it is hand-written.
