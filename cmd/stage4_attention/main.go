// Command stage4_attention prints the causal attention weights of a single
// untrained head over a short string, so you can see the pattern (each row sums
// to 1, upper triangle is zero). Run: go run ./cmd/stage4_attention
package main

import (
	"fmt"
	"math/rand"

	tiny "github.com/erubboli/go-tiny-llm"
)

func main() {
	const s = "hello"
	tok := tiny.NewTokenizer()
	ids := tok.Encode(s)

	rng := rand.New(rand.NewSource(tiny.RandSeed))
	emb := tiny.Param(tok.Size(), 8, rng)
	x := tiny.Embedding(emb, ids) // [T, 8]

	h := tiny.NewHead(8, 8, rng)
	w := h.Weights(x) // [T, T]

	fmt.Printf("attention weights over %q (row i = how char i attends):\n\n", s)
	fmt.Print("      ")
	for _, r := range s {
		fmt.Printf("%5c", r)
	}
	fmt.Println()
	runes := []rune(s)
	for i := range runes {
		fmt.Printf("%5c ", runes[i])
		for j := range runes {
			fmt.Printf("%5.2f", w.At(i, j))
		}
		fmt.Println()
	}
}
