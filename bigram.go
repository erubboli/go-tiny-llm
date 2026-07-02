package tiny

import "math/rand"

// Bigram is the simplest possible language model: a table counting how often
// each character follows each other character. No learning, no math, just
// tallies. It already "works", which is the surprise. Its flaw: it only ever
// looks one character back.
type Bigram struct {
	counts [][]float64 // counts[a][b] = times b followed a
	size   int
}

func NewBigram(vocabSize int) *Bigram {
	counts := make([][]float64, vocabSize)
	for i := range counts {
		counts[i] = make([]float64, vocabSize)
	}
	return &Bigram{counts: counts, size: vocabSize}
}

func (b *Bigram) Train(ids []int) {
	for i := 0; i+1 < len(ids); i++ {
		b.counts[ids[i]][ids[i+1]]++
	}
}

// Sample generates n characters starting from `start`, each drawn in proportion
// to how often it followed the previous character in the training text.
func (b *Bigram) Sample(rng *rand.Rand, start, n int) []int {
	out := make([]int, 0, n)
	cur := start
	for i := 0; i < n; i++ {
		nxt := b.sampleRow(rng, b.counts[cur])
		out = append(out, nxt)
		cur = nxt
	}
	return out
}

// sampleRow picks an index with probability proportional to its count.
func (b *Bigram) sampleRow(rng *rand.Rand, row []float64) int {
	total := 0.0
	for _, c := range row {
		total += c
	}
	if total == 0 {
		return rng.Intn(b.size) // unseen character: pick uniformly
	}
	r := rng.Float64() * total
	for i, c := range row {
		r -= c
		if r < 0 {
			return i
		}
	}
	return b.size - 1
}
