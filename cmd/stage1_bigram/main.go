// Command stage1_bigram trains the counting bigram model on data/input.txt and
// prints a sample. Run: go run ./cmd/stage1_bigram
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

	b := tiny.NewBigram(tok.Size())
	b.Train(ids)

	rng := rand.New(rand.NewSource(tiny.RandSeed))
	sample := b.Sample(rng, ids[0], 400)
	fmt.Println(tok.Decode(sample))
}
