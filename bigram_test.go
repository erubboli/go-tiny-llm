package tiny

import (
	"math/rand"
	"testing"
)

func TestBigramLearnsForcedPair(t *testing.T) {
	// Training data where 'a' is ALWAYS followed by 'b'. After counting,
	// sampling from 'a' must produce 'b'.
	tok := NewTokenizer()
	ids := tok.Encode("ababababab")
	b := NewBigram(tok.Size())
	b.Train(ids)

	rng := rand.New(rand.NewSource(1))
	aID := tok.Encode("a")[0]
	bID := tok.Encode("b")[0]
	out := b.Sample(rng, aID, 1) // one step after 'a'
	if out[0] != bID {
		t.Fatalf("after 'a' got id %d want 'b' id %d", out[0], bID)
	}
}
