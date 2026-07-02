package tiny

import (
	"math"
	"testing"
)

// gradCheck verifies analytic gradients against numerical ones. `build` returns
// a scalar (1x1) loss tensor given the current leaf data. `leaves` are the
// tensors whose .Grad we check. For each leaf entry we nudge it +/-eps, rebuild,
// and compare the numeric slope to the analytic grad.
func gradCheck(t *testing.T, leaves []*Tensor, build func() *Tensor) {
	const eps = 1e-5
	loss := build()
	loss.ZeroGrad()
	loss.Backward()

	for li, leaf := range leaves {
		for i := range leaf.Data {
			orig := leaf.Data[i]

			leaf.Data[i] = orig + eps
			plus := build().Data[0]
			leaf.Data[i] = orig - eps
			minus := build().Data[0]
			leaf.Data[i] = orig

			numeric := (plus - minus) / (2 * eps)
			analytic := leaf.Grad[i]
			if math.Abs(numeric-analytic) > 1e-4 {
				t.Fatalf("leaf %d entry %d: analytic %.6f numeric %.6f",
					li, i, analytic, numeric)
			}
		}
	}
}
