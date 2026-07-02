package tiny

import (
	"math/rand"
	"testing"
)

// The MLP must overfit a single (context -> target) example: loss goes near 0.
func TestMLPOverfitsOneExample(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	vocab, ctx, embed, hidden := 5, 3, 4, 8
	m := NewMLP(vocab, ctx, embed, hidden, rng)
	opt := NewAdamW(m.Params, 1e-2)

	context := []int{1, 2, 3}
	target := 4

	var last float64
	for i := 0; i < 300; i++ {
		loss := m.LossOn(context, target)
		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
		last = loss.Data[0]
	}
	if last > 0.05 {
		t.Fatalf("MLP failed to overfit one example: loss %.4f", last)
	}
}

func TestReshapeToRowGrad(t *testing.T) {
	a := Const(2, 3, []float64{1, 2, 3, 4, 5, 6})
	gradCheck(t, []*Tensor{a}, func() *Tensor { return reduce(reshapeToRow(a)) })
}
