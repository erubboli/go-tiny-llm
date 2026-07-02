package tiny

import "testing"

// reduce sums every entry of x into a 1x1 scalar, so gradCheck can drive it.
func reduce(x *Tensor) *Tensor {
	onesRight := Const(x.Cols, 1, ones(x.Cols))
	onesLeft := Const(1, x.Rows, ones(x.Rows))
	return MatMul(onesLeft, MatMul(x, onesRight))
}
func ones(n int) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = 1
	}
	return s
}

func TestReLUGrad(t *testing.T) {
	a := Const(2, 3, []float64{-1, 2, -3, 4, -5, 6})
	gradCheck(t, []*Tensor{a}, func() *Tensor { return reduce(ReLU(a)) })
}

func TestAddRowGrad(t *testing.T) {
	a := Const(3, 2, []float64{1, 2, 3, 4, 5, 6})
	row := Const(1, 2, []float64{7, 8})
	gradCheck(t, []*Tensor{a, row}, func() *Tensor { return reduce(AddRow(a, row)) })
}

func TestSoftmaxGrad(t *testing.T) {
	a := Const(2, 3, []float64{0.1, 0.2, 0.7, -1, 0, 2})
	// weight columns unequally so the gradient is non-trivial.
	w := Const(3, 1, []float64{1, 2, 3})
	onesL := Const(1, 2, []float64{1, 1})
	gradCheck(t, []*Tensor{a}, func() *Tensor {
		return MatMul(onesL, MatMul(Softmax(a), w))
	})
}

func TestRMSNormGrad(t *testing.T) {
	a := Const(2, 3, []float64{1, 2, 3, -1, 0.5, 2})
	g := Const(1, 3, []float64{1.1, 0.9, 1.0})
	w := Const(3, 1, []float64{1, 2, 3})
	onesL := Const(1, 2, []float64{1, 1})
	gradCheck(t, []*Tensor{a, g}, func() *Tensor {
		return MatMul(onesL, MatMul(RMSNorm(a, g), w))
	})
}

func TestTransposeGrad(t *testing.T) {
	a := Const(2, 3, []float64{1, 2, 3, 4, 5, 6})
	w := Const(2, 1, []float64{1, 2})
	onesL := Const(1, 3, []float64{1, 1, 1})
	gradCheck(t, []*Tensor{a}, func() *Tensor {
		return MatMul(onesL, MatMul(Transpose(a), w))
	})
}

func TestConcatColsGrad(t *testing.T) {
	a := Const(2, 2, []float64{1, 2, 3, 4})
	b := Const(2, 1, []float64{5, 6})
	gradCheck(t, []*Tensor{a, b}, func() *Tensor {
		return reduce(ConcatCols([]*Tensor{a, b}))
	})
}

func TestEmbeddingGrad(t *testing.T) {
	table := Const(4, 2, []float64{1, 2, 3, 4, 5, 6, 7, 8})
	ids := []int{0, 2, 2}
	gradCheck(t, []*Tensor{table}, func() *Tensor {
		return reduce(Embedding(table, ids))
	})
}

func TestCrossEntropyGrad(t *testing.T) {
	logits := Const(2, 3, []float64{0.5, 1.0, -0.5, 2.0, 0.0, 1.0})
	targets := []int{2, 0}
	gradCheck(t, []*Tensor{logits}, func() *Tensor {
		return CrossEntropy(logits, targets)
	})
}

func TestMaskedFillCausal(t *testing.T) {
	a := Const(2, 2, []float64{1, 2, 3, 4})
	out := MaskedFillCausal(a)
	if out.At(0, 1) > -1e8 {
		t.Fatalf("position (0,1) should be masked, got %v", out.At(0, 1))
	}
	if out.At(1, 0) != 3 {
		t.Fatalf("position (1,0) should be unchanged, got %v", out.At(1, 0))
	}
}

func TestMaskedFillCausalGrad(t *testing.T) {
	// Test the mask the way attention uses it: masked scores -> softmax.
	// Reducing the raw -1e9 sentinels directly causes catastrophic
	// cancellation in the finite-difference check, so compose through
	// Softmax (which maps -1e9 to ~0) and weight the columns for a
	// non-constant objective.
	a := Const(3, 3, []float64{0.1, 0.2, 0.3, -0.4, 0.5, 0.6, 0.7, -0.8, 0.9})
	w := Const(3, 1, []float64{1, 2, 3})
	onesL := Const(1, 3, []float64{1, 1, 1})
	gradCheck(t, []*Tensor{a}, func() *Tensor {
		return MatMul(onesL, MatMul(Softmax(MaskedFillCausal(a)), w))
	})
}
