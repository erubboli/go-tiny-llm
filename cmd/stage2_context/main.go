// Command stage2_context shows why we cannot just count longer contexts: the
// table size explodes. Run: go run ./cmd/stage2_context
package main

import (
	"fmt"

	tiny "github.com/erubboli/go-tiny-llm"
)

func main() {
	tok := tiny.NewTokenizer()
	v := tok.Size()
	fmt.Printf("vocabulary: %d characters\n\n", v)
	fmt.Println("context | table rows needed")
	for _, n := range []int{1, 2, 3, 5, 10, 20} {
		fmt.Printf("%7d | %.3g\n", n, tiny.NGramTableSize(v, n))
	}
	fmt.Println("\nA 20-character context needs more rows than there are atoms")
	fmt.Println("in your body. Counting is a dead end. The model must LEARN.")
}
