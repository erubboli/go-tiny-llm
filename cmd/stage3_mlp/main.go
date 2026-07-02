// Command stage3_mlp trains the learning MLP on data/input.txt with a small
// fixed context and prints the loss falling. Run: go run ./cmd/stage3_mlp
package main

import (
	"fmt"
	"math/rand"
	"os"

	tiny "github.com/erubboli/go-tiny-llm"
)

func main() {
	const ctx, embed, hidden = 3, 16, 64
	text, err := os.ReadFile("data/input.txt")
	if err != nil {
		fmt.Println("read data/input.txt:", err)
		os.Exit(1)
	}
	tok := tiny.NewTokenizer()
	ids := tok.Encode(string(text))

	rng := rand.New(rand.NewSource(tiny.RandSeed))
	m := tiny.NewMLP(tok.Size(), ctx, embed, hidden, rng)
	opt := tiny.NewAdamW(m.Params, tiny.LearningRate)

	for step := 0; step < 2000; step++ {
		i := rng.Intn(len(ids) - ctx - 1)
		loss := m.LossOn(ids[i:i+ctx], ids[i+ctx])
		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
		if step%200 == 0 {
			fmt.Printf("step %4d  loss %.4f\n", step, loss.Data[0])
		}
	}
}
