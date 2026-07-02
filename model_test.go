package tiny

import (
	"math/rand"
	"testing"
)

// The full model must reduce loss on a tiny repeating pattern within a few
// hundred steps — proof the whole stack (embeddings, attention, blocks, loss,
// backprop, AdamW) is wired correctly end to end.
func TestModelLearnsPattern(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	tok := NewTokenizer()
	ids := tok.Encode("abcabcabcabcabcabcabcabc")

	m := NewModel(tok.Size(), rng)
	opt := NewAdamW(m.Params, 1e-3)

	seq := 8
	first := 0.0
	var last float64
	for step := 0; step < 400; step++ {
		i := rng.Intn(len(ids) - seq)
		loss := m.Loss(ids[i:i+seq], ids[i+1:i+seq+1])
		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
		if step == 0 {
			first = loss.Data[0]
		}
		last = loss.Data[0]
	}
	if last >= first {
		t.Fatalf("model did not learn: first %.3f last %.3f", first, last)
	}
}
