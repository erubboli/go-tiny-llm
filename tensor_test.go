package tiny

import "testing"

func TestMatMulForward(t *testing.T) {
	a := Const(1, 2, []float64{1, 2})
	b := Const(2, 1, []float64{3, 4})
	c := MatMul(a, b) // 1x1 = 1*3 + 2*4 = 11
	if c.Data[0] != 11 {
		t.Fatalf("matmul got %v want 11", c.Data[0])
	}
}

func TestMatMulGrad(t *testing.T) {
	a := Const(2, 3, []float64{1, 2, 3, 4, 5, 6})
	b := Const(3, 2, []float64{6, 5, 4, 3, 2, 1})
	// Reduce a@b (2x2) to a scalar so gradCheck can drive it.
	ones := Const(2, 1, []float64{1, 1})
	onesT := Const(1, 2, []float64{1, 1})
	build := func() *Tensor {
		return MatMul(onesT, MatMul(MatMul(a, b), ones)) // 1x1 scalar
	}
	gradCheck(t, []*Tensor{a, b}, build)
}

func TestAddGrad(t *testing.T) {
	a := Const(2, 2, []float64{1, 2, 3, 4})
	b := Const(2, 2, []float64{5, 6, 7, 8})
	ones := Const(2, 1, []float64{1, 1})
	onesT := Const(1, 2, []float64{1, 1})
	build := func() *Tensor { return MatMul(onesT, MatMul(Add(a, b), ones)) }
	gradCheck(t, []*Tensor{a, b}, build)
}

func TestMulScalarGrad(t *testing.T) {
	a := Const(2, 2, []float64{1, 2, 3, 4})
	ones := Const(2, 1, []float64{1, 1})
	onesT := Const(1, 2, []float64{1, 1})
	build := func() *Tensor {
		return MatMul(onesT, MatMul(MulScalar(a, 3.0), ones))
	}
	gradCheck(t, []*Tensor{a}, build)
}
