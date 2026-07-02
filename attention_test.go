package tiny

import (
	"math"
	"math/rand"
	"testing"
)

func TestHeadShapeAndCausalGrad(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	h := NewHead(4, 4, rng)
	x := Const(3, 4, []float64{
		1, 0, 0, 1,
		0, 1, 1, 0,
		1, 1, 0, 0,
	})
	out := h.Forward(x)
	if out.Rows != 3 || out.Cols != 4 {
		t.Fatalf("head out shape %dx%d want 3x4", out.Rows, out.Cols)
	}
	// Gradient must flow back to x (end-to-end differentiability).
	gradCheck(t, []*Tensor{x}, func() *Tensor { return reduce(h.Forward(x)) })
}

func TestMultiHeadShape(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	mh := NewMultiHead(8, 4, 2, rng) // embed 8 = 4 heads x 2
	x := Const(5, 8, make([]float64, 40))
	for i := range x.Data {
		x.Data[i] = math.Sin(float64(i))
	}
	out := mh.Forward(x)
	if out.Rows != 5 || out.Cols != 8 {
		t.Fatalf("multihead out shape %dx%d want 5x8", out.Rows, out.Cols)
	}
}
