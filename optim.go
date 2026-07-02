package tiny

import (
	"math"
	"math/rand"
)

// Param creates a trainable tensor with small random values. Scale ~1/sqrt(fan)
// keeps early activations from blowing up.
func Param(rows, cols int, rng *rand.Rand) *Tensor {
	t := NewTensor(rows, cols)
	scale := 1.0 / math.Sqrt(float64(cols))
	for i := range t.Data {
		t.Data[i] = rng.NormFloat64() * scale
	}
	return t
}

// AdamW is the optimizer: it remembers a running average of each gradient and
// its square, giving every parameter its own adaptive step size. Plain gradient
// descent struggles to train transformers; Adam is what makes it converge.
// The "W" stands for decoupled weight decay, applied separately from the
// gradient-based Adam step (see WeightDecay constant).
type AdamW struct {
	params []*Tensor
	lr     float64
	m, v   [][]float64 // first and second moment per param
	t      int         // step count, for bias correction
}

func NewAdamW(params []*Tensor, lr float64) *AdamW {
	m := make([][]float64, len(params))
	v := make([][]float64, len(params))
	for i, p := range params {
		m[i] = make([]float64, len(p.Data))
		v[i] = make([]float64, len(p.Data))
	}
	return &AdamW{params: params, lr: lr, m: m, v: v}
}

func (o *AdamW) Step() {
	o.t++
	b1c := 1 - math.Pow(AdamBeta1, float64(o.t))
	b2c := 1 - math.Pow(AdamBeta2, float64(o.t))
	for pi, p := range o.params {
		for i := range p.Data {
			g := p.Grad[i]
			o.m[pi][i] = AdamBeta1*o.m[pi][i] + (1-AdamBeta1)*g
			o.v[pi][i] = AdamBeta2*o.v[pi][i] + (1-AdamBeta2)*g*g
			mhat := o.m[pi][i] / b1c
			vhat := o.v[pi][i] / b2c
			// AdamW: decoupled weight decay added directly to the update,
			// separate from the gradient-based Adam step.
			p.Data[i] -= o.lr * (mhat/(math.Sqrt(vhat)+AdamEps) + WeightDecay*p.Data[i])
		}
	}
}

func (o *AdamW) ZeroGrad() {
	for _, p := range o.params {
		for i := range p.Grad {
			p.Grad[i] = 0
		}
	}
}
