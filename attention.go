package tiny

import (
	"math"
	"math/rand"
)

// Head is one attention head. Each position produces a query, a key, and a
// value. A position's output is a weighted average of the values of all
// positions it is allowed to see (itself and earlier — the causal mask), where
// the weights come from how well its query matches each key.
type Head struct {
	Wq, Wk, Wv *Tensor // each [embed, headSize]
	headSize   int
}

func NewHead(embed, headSize int, rng *rand.Rand) *Head {
	return &Head{
		Wq:       Param(embed, headSize, rng),
		Wk:       Param(embed, headSize, rng),
		Wv:       Param(embed, headSize, rng),
		headSize: headSize,
	}
}

func (h *Head) Forward(x *Tensor) *Tensor {
	v := MatMul(x, h.Wv)          // [T, hs]
	return MatMul(h.Weights(x), v) // [T, hs]
}

// Weights returns the causal softmax attention weights [T,T] for input x.
// Exposed so the stage-4 demo can visualise them.
func (h *Head) Weights(x *Tensor) *Tensor {
	q := MatMul(x, h.Wq) // [T, hs]
	k := MatMul(x, h.Wk) // [T, hs]
	// scores[i][j] = q_i . k_j, scaled so softmax stays well-behaved.
	scores := MulScalar(MatMul(q, Transpose(k)), 1/math.Sqrt(float64(h.headSize)))
	scores = MaskedFillCausal(scores) // no peeking at the future
	return Softmax(scores)            // [T, T], each row sums to 1
}

func (h *Head) Params() []*Tensor { return []*Tensor{h.Wq, h.Wk, h.Wv} }

// MultiHead runs several heads in parallel and mixes their outputs. Different
// heads can specialise (one tracks the previous letter, another the start of
// the word, etc.).
type MultiHead struct {
	heads []*Head
	proj  *Tensor // [embed, embed], mixes concatenated head outputs
}

func NewMultiHead(embed, numHeads, headSize int, rng *rand.Rand) *MultiHead {
	heads := make([]*Head, numHeads)
	for i := range heads {
		heads[i] = NewHead(embed, headSize, rng)
	}
	return &MultiHead{heads: heads, proj: Param(embed, embed, rng)}
}

func (mh *MultiHead) Forward(x *Tensor) *Tensor {
	outs := make([]*Tensor, len(mh.heads))
	for i, h := range mh.heads {
		outs[i] = h.Forward(x)
	}
	concat := ConcatCols(outs) // [T, numHeads*headSize] == [T, embed]
	return MatMul(concat, mh.proj)
}

func (mh *MultiHead) Params() []*Tensor {
	var p []*Tensor
	for _, h := range mh.heads {
		p = append(p, h.Params()...)
	}
	return append(p, mh.proj)
}
