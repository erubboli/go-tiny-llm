package tiny

import (
	"math/rand"
	"testing"
)

// Fit a tiny linear layer y = x @ W to fixed targets with AdamW; loss must drop.
func TestAdamWReducesLoss(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	W := Param(2, 2, rng)
	x := Const(3, 2, []float64{1, 0, 0, 1, 1, 1})
	targets := []int{0, 1, 0} // arbitrary but fixed

	opt := NewAdamW([]*Tensor{W}, 1e-1)

	first := lossOnce(W, x, targets, opt) // also does one step
	var last float64
	for i := 0; i < 200; i++ {
		last = lossOnce(W, x, targets, opt)
	}
	if last >= first {
		t.Fatalf("loss did not decrease: first %.4f last %.4f", first, last)
	}
}

func lossOnce(W, x *Tensor, targets []int, opt *AdamW) float64 {
	logits := MatMul(x, W)
	loss := CrossEntropy(logits, targets)
	opt.ZeroGrad()
	loss.Backward()
	opt.Step()
	return loss.Data[0]
}
