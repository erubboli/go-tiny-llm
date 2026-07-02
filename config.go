// Package tiny is a from-scratch, character-level transformer language model.
// Everything here is written for clarity, not speed. If a loop is obvious and
// slow, that is the intended trade: you should be able to read the whole model.
package tiny

// Vocab is every character the model can read or write. Anything outside this
// set is dropped when we clean training data. ~70 characters => tiny model.
const Vocab = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.!?'-\n"

const (
	// ContextWindow is how many characters the model can look back at.
	// This is the single number people mean by "context length".
	// Lower it (e.g. 64) to train much faster while experimenting.
	ContextWindow = 128

	// EmbedDim is the width of the vectors that represent each character.
	EmbedDim = 64

	NumLayers = 2 // transformer blocks stacked on top of each other
	NumHeads  = 4 // attention heads per block
	HeadSize  = EmbedDim / NumHeads

	FeedForwardMult = 4 // hidden layer of each block is 4x wider than EmbedDim

	LearningRate = 3e-4
	TrainSteps   = 3000
	SampleLen    = 400 // characters to generate when we show off

	RandSeed = 1337

	RMSNormEps = 1e-5
	AdamBeta1  = 0.9
	AdamBeta2  = 0.99
	AdamEps    = 1e-8
)
