// Command stage5_train trains the full tiny transformer on data/input.txt,
// prints the loss falling, and generates a sample. Run: go run ./cmd/stage5_train
package main

import (
	"fmt"
	"math/rand"
	"os"

	tiny "github.com/erubboli/go-tiny-llm"
)

func main() {
	text, err := os.ReadFile("data/input.txt")
	if err != nil {
		fmt.Println("read data/input.txt:", err)
		os.Exit(1)
	}
	tok := tiny.NewTokenizer()
	ids := tok.Encode(string(text))
	if len(ids) < tiny.ContextWindow+2 {
		fmt.Println("training text too short")
		os.Exit(1)
	}

	rng := rand.New(rand.NewSource(tiny.RandSeed))
	m := tiny.NewModel(tok.Size(), rng)
	opt := tiny.NewAdamW(m.Params, tiny.LearningRate)

	seq := tiny.ContextWindow
	for step := 0; step < tiny.TrainSteps; step++ {
		i := rng.Intn(len(ids) - seq)
		loss := m.Loss(ids[i:i+seq], ids[i+1:i+seq+1])
		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
		if step%100 == 0 {
			fmt.Printf("step %5d/%d  loss %.4f\n", step, tiny.TrainSteps, loss.Data[0])
		}
	}

	fmt.Println("\n--- sample ---")
	seed := ids[:1]
	gen := m.Generate(rng, seed, tiny.SampleLen)
	fmt.Println(tok.Decode(gen))
}
