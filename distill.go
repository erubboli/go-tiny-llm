package tiny

import "math"

// DistillLoss is the training loss for knowledge distillation (Hinton, Vinyals
// and Dean, 2015): the student's logits are scored against the true next
// characters AND against a teacher's softened opinion of the same positions,
// and the two terms are added.
//
// temperature > 1 softens both distributions before they are compared. A
// confident teacher puts almost all of its probability on one character, which
// hides what it has learned about the runners-up; dividing the logits by the
// temperature before the softmax brings that information back (the "dark
// knowledge" of the original paper). The soft term is scaled by temperature^2
// so its gradient stays comparable across temperatures.
//
// alpha weighs the hard loss; (1-alpha) weighs the soft one. alpha = 1 is
// plain CrossEntropy. teacherLogits must already be Detached: gradients stop
// at the student, never flow into the teacher.
func DistillLoss(studentLogits, teacherLogits *Tensor, targets []int, temperature, alpha float64) *Tensor {
	if studentLogits.Rows != teacherLogits.Rows || studentLogits.Cols != teacherLogits.Cols {
		panic("distill shape mismatch")
	}
	if temperature <= 0 {
		panic("temperature must be positive")
	}
	T, V := studentLogits.Rows, studentLogits.Cols

	// Hard term: the ordinary next-character loss, weighted by alpha.
	hard := MulScalar(CrossEntropy(studentLogits, targets), alpha)

	// Soft term: KL(teacher || student) between temperature-softened
	// distributions, averaged over positions, scaled by T^2 and (1-alpha).
	p := make([]float64, T*V)
	q := make([]float64, T*V)
	var loss float64
	for i := 0; i < T; i++ {
		off := i * V
		softmaxRow(teacherLogits.Data[off:off+V], temperature, p[off:off+V])
		softmaxRow(studentLogits.Data[off:off+V], temperature, q[off:off+V])
		for j := 0; j < V; j++ {
			if p[off+j] > 1e-12 {
				loss += p[off+j] * (math.Log(p[off+j]) - math.Log(q[off+j]+1e-12))
			}
		}
	}
	loss *= (1 - alpha) * temperature * temperature / float64(T)

	soft := NewTensor(1, 1)
	soft.Data[0] = loss
	soft.parents = []*Tensor{studentLogits}
	soft.backward = func() {
		// d KL / d student logit = (q - p) / temperature, by the same algebra
		// that makes CrossEntropy's gradient (softmax - onehot): the teacher
		// distribution is a constant here, so only the q side contributes.
		scale := soft.Grad[0] * (1 - alpha) * temperature / float64(T)
		for i := 0; i < T; i++ {
			off := i * V
			for j := 0; j < V; j++ {
				studentLogits.Grad[off+j] += (q[off+j] - p[off+j]) * scale
			}
		}
	}
	return Add(hard, soft)
}

// softmaxRow writes the softmax of one row of raw logits, divided by a
// temperature first, into out. Bigger temperature, flatter distribution.
func softmaxRow(logits []float64, temperature float64, out []float64) {
	max := math.Inf(-1)
	for j, v := range logits {
		v /= temperature
		out[j] = v
		if v > max {
			max = v
		}
	}
	var sum float64
	for j := range out {
		out[j] = math.Exp(out[j] - max)
		sum += out[j]
	}
	for j := range out {
		out[j] /= sum
	}
}

// Detach returns a leaf copy of t: the same numbers, but no parents, so a
// backward pass that reaches it stops instead of flowing into whatever
// computed t. This is how a frozen teacher donates logits to a student.
func Detach(t *Tensor) *Tensor {
	return Const(t.Rows, t.Cols, append([]float64(nil), t.Data...))
}
