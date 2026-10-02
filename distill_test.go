package tiny

import (
	"math"
	"math/rand"
	"testing"
)

func distillFixture(rng *rand.Rand) (student, teacher *Tensor, targets []int) {
	const T, V = 8, 13
	student = Param(T, V, rng)
	teacher = Detach(Param(T, V, rng))
	targets = make([]int, T)
	for i := range targets {
		targets[i] = rng.Intn(V)
	}
	return student, teacher, targets
}

func manualSoftKL(p, q []float64, rows int, temperature float64) float64 {
	V := len(p) / rows
	var kl float64
	for i := 0; i < rows; i++ {
		pSoft := make([]float64, V)
		qSoft := make([]float64, V)
		softmaxRow(p[i*V:(i+1)*V], temperature, pSoft)
		softmaxRow(q[i*V:(i+1)*V], temperature, qSoft)
		for j := range pSoft {
			if pSoft[j] > 1e-12 {
				kl += pSoft[j] * (math.Log(pSoft[j]) - math.Log(qSoft[j]+1e-12))
			}
		}
	}
	return kl
}

func TestDistillLossAlphaOneIsCrossEntropy(t *testing.T) {
	s, teacher, targets := distillFixture(rand.New(rand.NewSource(1)))
	got := DistillLoss(s, teacher, targets, 4.0, 1.0).Data[0]
	want := CrossEntropy(s, targets).Data[0]
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("alpha=1 should be plain CrossEntropy: got %.10f want %.10f", got, want)
	}
}

func TestDistillLossAlphaZeroIsScaledKL(t *testing.T) {
	s, teacher, targets := distillFixture(rand.New(rand.NewSource(2)))
	const temperature = 3.0
	got := DistillLoss(s, teacher, targets, temperature, 0.0).Data[0]
	want := manualSoftKL(teacher.Data, s.Data, s.Rows, temperature) * temperature * temperature / float64(s.Rows)
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("alpha=0 should be T^2 * KL: got %.10f want %.10f", got, want)
	}
}

func TestDistillLossGradCheck(t *testing.T) {
	s, teacher, targets := distillFixture(rand.New(rand.NewSource(3)))
	gradCheck(t, []*Tensor{s}, func() *Tensor {
		return DistillLoss(s, teacher, targets, 3.0, 0.5)
	})
}

func TestDetachStopsGradient(t *testing.T) {
	inner := Param(2, 2, rand.New(rand.NewSource(4)))
	d := Detach(inner)
	reduce(d).Backward()
	for _, g := range inner.Grad {
		if g != 0 {
			t.Fatalf("gradient leaked through Detach: %v", inner.Grad)
		}
	}
}
