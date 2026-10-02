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
	return NewBlockDims(EmbedDim, NumHeads, rng)
}

// NewBlockDims builds a block of a given width. The feed-forward hidden layer
// stays at 4x the embedding dimension, as in the default model.
func NewBlockDims(embedDim, numHeads int, rng *rand.Rand) *Block {
	hidden := embedDim * FeedForwardMult
	return &Block{
		norm1: onesRow(embedDim),
		norm2: onesRow(embedDim),
		attn:  NewMultiHead(embedDim, numHeads, embedDim/numHeads, rng),
		w1:    Param(embedDim, hidden, rng),
		b1:    NewTensor(1, hidden),
		w2:    Param(hidden, embedDim, rng),
		b2:    NewTensor(1, embedDim),
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
	posEmb *Tensor // [contextWindow, embed]
	blocks []*Block
	normF  *Tensor // [1, embed]
	head   *Tensor // [embed, vocab]

	ctxWindow int // this model's own context window (normally ContextWindow)
	Params    []*Tensor
}

// ModelConfig describes the shape of a Model. Zero fields fall back to the
// package constants, so DefaultModelConfig reproduces NewModel exactly.
type ModelConfig struct {
	EmbedDim      int
	NumLayers     int
	NumHeads      int
	ContextWindow int
}

// DefaultModelConfig is the shape trained by stage5_train.
func DefaultModelConfig() ModelConfig {
	return ModelConfig{
		EmbedDim:      EmbedDim,
		NumLayers:     NumLayers,
		NumHeads:      NumHeads,
		ContextWindow: ContextWindow,
	}
}

func NewModel(vocabSize int, rng *rand.Rand) *Model {
	return NewModelConfig(vocabSize, DefaultModelConfig(), rng)
}

// NewModelConfig builds a model of an arbitrary shape. Distillation uses this
// to build a student narrower and shallower than its teacher.
func NewModelConfig(vocabSize int, cfg ModelConfig, rng *rand.Rand) *Model {
	if cfg.EmbedDim == 0 {
		cfg.EmbedDim = EmbedDim
	}
	if cfg.NumLayers == 0 {
		cfg.NumLayers = NumLayers
	}
	if cfg.NumHeads == 0 {
		cfg.NumHeads = NumHeads
	}
	if cfg.ContextWindow == 0 {
		cfg.ContextWindow = ContextWindow
	}
	m := &Model{
		tokEmb:    Param(vocabSize, cfg.EmbedDim, rng),
		posEmb:    Param(cfg.ContextWindow, cfg.EmbedDim, rng),
		normF:     onesRow(cfg.EmbedDim),
		head:      Param(cfg.EmbedDim, vocabSize, rng),
		ctxWindow: cfg.ContextWindow,
	}
	m.blocks = make([]*Block, cfg.NumLayers)
	for i := range m.blocks {
		m.blocks[i] = NewBlockDims(cfg.EmbedDim, cfg.NumHeads, rng)
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
// append, repeat, feeding at most the last m.ctxWindow characters back in.
func (m *Model) Generate(rng *rand.Rand, seed []int, n int) []int {
	out := append([]int{}, seed...)
	for i := 0; i < n; i++ {
		ctx := out
		if len(ctx) > m.ctxWindow {
			ctx = ctx[len(ctx)-m.ctxWindow:]
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
