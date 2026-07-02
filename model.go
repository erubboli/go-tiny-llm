package tiny

import "math/rand"

// Block is one transformer layer: attention lets positions share information,
// then a small per-position feed-forward network digests it. Both are wrapped
// in a residual connection (x + sublayer(norm(x))) so gradients flow easily and
// each layer only has to learn a refinement.
type Block struct {
	norm1, norm2 *Tensor // RMSNorm gains, [1, embed]
	attn         *MultiHead
	w1, b1       *Tensor // feed-forward: [embed, hidden], [1, hidden]
	w2, b2       *Tensor // [hidden, embed], [1, embed]
}

func NewBlock(rng *rand.Rand) *Block {
	hidden := EmbedDim * FeedForwardMult
	return &Block{
		norm1: onesRow(EmbedDim),
		norm2: onesRow(EmbedDim),
		attn:  NewMultiHead(EmbedDim, NumHeads, HeadSize, rng),
		w1:    Param(EmbedDim, hidden, rng),
		b1:    NewTensor(1, hidden),
		w2:    Param(hidden, EmbedDim, rng),
		b2:    NewTensor(1, EmbedDim),
	}
}

func (b *Block) Forward(x *Tensor) *Tensor {
	x = Add(x, b.attn.Forward(RMSNorm(x, b.norm1)))
	ff := AddRow(MatMul(ReLU(AddRow(MatMul(RMSNorm(x, b.norm2), b.w1), b.b1)), b.w2), b.b2)
	return Add(x, ff)
}

func (b *Block) Params() []*Tensor {
	p := []*Tensor{b.norm1, b.norm2, b.w1, b.b1, b.w2, b.b2}
	return append(p, b.attn.Params()...)
}

// Model is the whole thing: character + position embeddings, a stack of blocks,
// a final norm, and a projection to a score per possible next character.
type Model struct {
	tokEmb *Tensor // [vocab, embed]
	posEmb *Tensor // [ContextWindow, embed]
	blocks []*Block
	normF  *Tensor // [1, embed]
	head   *Tensor // [embed, vocab]
	Params []*Tensor
}

func NewModel(vocabSize int, rng *rand.Rand) *Model {
	m := &Model{
		tokEmb: Param(vocabSize, EmbedDim, rng),
		posEmb: Param(ContextWindow, EmbedDim, rng),
		normF:  onesRow(EmbedDim),
		head:   Param(EmbedDim, vocabSize, rng),
	}
	m.blocks = make([]*Block, NumLayers)
	for i := range m.blocks {
		m.blocks[i] = NewBlock(rng)
	}
	m.Params = []*Tensor{m.tokEmb, m.posEmb, m.normF, m.head}
	for _, b := range m.blocks {
		m.Params = append(m.Params, b.Params()...)
	}
	return m
}

// Forward returns [T, vocab] logits: a next-character score distribution for
// every position in the input.
func (m *Model) Forward(ids []int) *Tensor {
	T := len(ids)
	pos := make([]int, T)
	for i := range pos {
		pos[i] = i // positions 0,1,2,...
	}
	x := Add(Embedding(m.tokEmb, ids), Embedding(m.posEmb, pos)) // [T, embed]
	for _, b := range m.blocks {
		x = b.Forward(x)
	}
	x = RMSNorm(x, m.normF)
	return MatMul(x, m.head) // [T, vocab]
}

// Loss compares the model's prediction at every position to the actual next
// character (targets = ids shifted by one).
func (m *Model) Loss(ids, targets []int) *Tensor {
	return CrossEntropy(m.Forward(ids), targets)
}

// Generate autoregressively extends `seed` by n characters: predict, sample,
// append, repeat, feeding at most the last ContextWindow characters back in.
func (m *Model) Generate(rng *rand.Rand, seed []int, n int) []int {
	out := append([]int{}, seed...)
	for i := 0; i < n; i++ {
		ctx := out
		if len(ctx) > ContextWindow {
			ctx = ctx[len(ctx)-ContextWindow:]
		}
		logits := m.Forward(ctx)
		// take the last row (prediction for the next character)
		last := Const(1, logits.Cols,
			logits.Data[(logits.Rows-1)*logits.Cols:logits.Rows*logits.Cols])
		probs := Softmax(last)
		out = append(out, sampleFromRow(rng, probs.Data))
	}
	return out
}

func sampleFromRow(rng *rand.Rand, p []float64) int {
	r := rng.Float64()
	for i, pi := range p {
		r -= pi
		if r < 0 {
			return i
		}
	}
	return len(p) - 1
}

// onesRow makes a trainable [1,n] gain vector initialised to 1 (for RMSNorm).
func onesRow(n int) *Tensor {
	t := NewTensor(1, n)
	for i := range t.Data {
		t.Data[i] = 1
	}
	return t
}
