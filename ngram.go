package tiny

import "math"

// NGramTableSize returns how many rows a full lookup table would need to
// remember the previous n characters: vocabSize^n. Returned as float64 because
// the true count overflows int64 almost immediately — which is exactly the
// point. Counting cannot scale to real context; the model must learn instead.
func NGramTableSize(vocabSize, n int) float64 {
	return math.Pow(float64(vocabSize), float64(n))
}
