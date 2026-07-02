package tiny

import "math/rand"

// MLP is the first model that LEARNS rather than counts. It embeds a small,
// fixed context of characters, flattens them into one long vector, passes it
// through one hidden ReLU layer, and projects to a score per possible next
// character. All weights are trained by gradient descent.
type MLP struct {
	emb    *Tensor // [vocab, embed]
	w1, b1 *Tensor // [ctx*embed, hidden], [1, hidden]
	w2, b2 *Tensor // [hidden, vocab], [1, vocab]
	ctx    int
	embed  int
	Params []*Tensor
}

func NewMLP(vocabSize, ctx, embed, hidden int, rng *rand.Rand) *MLP {
	m := &MLP{
		emb:   Param(vocabSize, embed, rng),
		w1:    Param(ctx*embed, hidden, rng),
		b1:    NewTensor(1, hidden),
		w2:    Param(hidden, vocabSize, rng),
		b2:    NewTensor(1, vocabSize),
		ctx:   ctx,
		embed: embed,
	}
	m.Params = []*Tensor{m.emb, m.w1, m.b1, m.w2, m.b2}
	return m
}

// Forward returns [1, vocab] logits for the character following `ids`
// (len(ids) must equal ctx).
func (m *MLP) Forward(ids []int) *Tensor {
	// Embed each context char, then flatten to a single [1, ctx*embed] row.
	rows := Embedding(m.emb, ids)                    // [ctx, embed]
	flat := reshapeToRow(rows)                       // [1, ctx*embed]
	hidden := ReLU(AddRow(MatMul(flat, m.w1), m.b1)) // [1, hidden]
	return AddRow(MatMul(hidden, m.w2), m.b2)        // [1, vocab]
}

func (m *MLP) LossOn(context []int, target int) *Tensor {
	return CrossEntropy(m.Forward(context), []int{target})
}

// reshapeToRow flattens [r,c] into [1, r*c] as an autograd op.
func reshapeToRow(a *Tensor) *Tensor {
	out := NewTensor(1, a.Rows*a.Cols)
	copy(out.Data, a.Data)
	out.parents = []*Tensor{a}
	out.backward = func() {
		for i := range a.Grad {
			a.Grad[i] += out.Grad[i]
		}
	}
	return out
}
