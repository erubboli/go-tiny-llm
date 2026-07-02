package tiny

import "math"

// AddRow adds a [1,C] bias vector to every row of a [T,C] matrix.
func AddRow(a, row *Tensor) *Tensor {
	out := NewTensor(a.Rows, a.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Cols; j++ {
			out.Data[i*a.Cols+j] = a.Data[i*a.Cols+j] + row.Data[j]
		}
	}
	out.parents = []*Tensor{a, row}
	out.backward = func() {
		for i := 0; i < a.Rows; i++ {
			for j := 0; j < a.Cols; j++ {
				g := out.Grad[i*a.Cols+j]
				a.Grad[i*a.Cols+j] += g
				row.Grad[j] += g // broadcast: sum grad over rows
			}
		}
	}
	return out
}

// ReLU zeroes negatives. Gradient flows only where the input was positive.
func ReLU(a *Tensor) *Tensor {
	out := NewTensor(a.Rows, a.Cols)
	for i := range a.Data {
		if a.Data[i] > 0 {
			out.Data[i] = a.Data[i]
		}
	}
	out.parents = []*Tensor{a}
	out.backward = func() {
		for i := range a.Data {
			if a.Data[i] > 0 {
				a.Grad[i] += out.Grad[i]
			}
		}
	}
	return out
}

// Softmax turns each row into a probability distribution.
func Softmax(a *Tensor) *Tensor {
	out := NewTensor(a.Rows, a.Cols)
	for i := 0; i < a.Rows; i++ {
		off := i * a.Cols
		max := math.Inf(-1)
		for j := 0; j < a.Cols; j++ {
			if a.Data[off+j] > max {
				max = a.Data[off+j]
			}
		}
		var sum float64
		for j := 0; j < a.Cols; j++ {
			e := math.Exp(a.Data[off+j] - max)
			out.Data[off+j] = e
			sum += e
		}
		for j := 0; j < a.Cols; j++ {
			out.Data[off+j] /= sum
		}
	}
	out.parents = []*Tensor{a}
	out.backward = func() {
		// For each row: dx_i = y_i * (dy_i - sum_j dy_j y_j)
		for i := 0; i < a.Rows; i++ {
			off := i * a.Cols
			var dot float64
			for j := 0; j < a.Cols; j++ {
				dot += out.Grad[off+j] * out.Data[off+j]
			}
			for j := 0; j < a.Cols; j++ {
				y := out.Data[off+j]
				a.Grad[off+j] += y * (out.Grad[off+j] - dot)
			}
		}
	}
	return out
}

// RMSNorm scales each row to unit root-mean-square, then multiplies per-feature
// by gamma ([1,C]). Simpler than LayerNorm (no mean subtraction); this is what
// modern models like Llama use.
func RMSNorm(a, gamma *Tensor) *Tensor {
	out := NewTensor(a.Rows, a.Cols)
	C := a.Cols
	r := make([]float64, a.Rows) // per-row 1/sqrt(mean(x^2)+eps)
	for i := 0; i < a.Rows; i++ {
		off := i * C
		var ms float64
		for j := 0; j < C; j++ {
			ms += a.Data[off+j] * a.Data[off+j]
		}
		ms /= float64(C)
		r[i] = 1 / math.Sqrt(ms+RMSNormEps)
		for j := 0; j < C; j++ {
			out.Data[off+j] = a.Data[off+j] * r[i] * gamma.Data[j]
		}
	}
	out.parents = []*Tensor{a, gamma}
	out.backward = func() {
		for i := 0; i < a.Rows; i++ {
			off := i * C
			// s = sum_k g_k * gamma_k * x_k
			var s float64
			for k := 0; k < C; k++ {
				s += out.Grad[off+k] * gamma.Data[k] * a.Data[off+k]
			}
			r3 := r[i] * r[i] * r[i]
			for j := 0; j < C; j++ {
				g := out.Grad[off+j]
				// dx_j = r*g_j*gamma_j - (x_j r^3 / C) * s
				a.Grad[off+j] += r[i]*g*gamma.Data[j] - (a.Data[off+j]*r3/float64(C))*s
				// dgamma_j += g_j * x_j * r
				gamma.Grad[j] += g * a.Data[off+j] * r[i]
			}
		}
	}
	return out
}

// MaskedFillCausal sets entry (i,j) to -1e9 when j > i, so after softmax a
// position cannot attend to the future. Masked entries get zero gradient.
func MaskedFillCausal(a *Tensor) *Tensor {
	out := NewTensor(a.Rows, a.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Cols; j++ {
			if j > i {
				out.Data[i*a.Cols+j] = -1e9
			} else {
				out.Data[i*a.Cols+j] = a.Data[i*a.Cols+j]
			}
		}
	}
	out.parents = []*Tensor{a}
	out.backward = func() {
		for i := 0; i < a.Rows; i++ {
			for j := 0; j <= i && j < a.Cols; j++ {
				a.Grad[i*a.Cols+j] += out.Grad[i*a.Cols+j]
			}
		}
	}
	return out
}

// Transpose swaps rows and columns.
func Transpose(a *Tensor) *Tensor {
	out := NewTensor(a.Cols, a.Rows)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Cols; j++ {
			out.Data[j*a.Rows+i] = a.Data[i*a.Cols+j]
		}
	}
	out.parents = []*Tensor{a}
	out.backward = func() {
		for i := 0; i < a.Rows; i++ {
			for j := 0; j < a.Cols; j++ {
				a.Grad[i*a.Cols+j] += out.Grad[j*a.Rows+i]
			}
		}
	}
	return out
}

// ConcatCols glues matrices side by side (same row count). Used to join the
// outputs of several attention heads back into one [T,C] matrix.
func ConcatCols(parts []*Tensor) *Tensor {
	rows := parts[0].Rows
	total := 0
	for _, p := range parts {
		total += p.Cols
	}
	out := NewTensor(rows, total)
	colOff := 0
	for _, p := range parts {
		for i := 0; i < rows; i++ {
			for j := 0; j < p.Cols; j++ {
				out.Data[i*total+colOff+j] = p.Data[i*p.Cols+j]
			}
		}
		colOff += p.Cols
	}
	out.parents = append([]*Tensor{}, parts...)
	out.backward = func() {
		colOff := 0
		for _, p := range parts {
			for i := 0; i < rows; i++ {
				for j := 0; j < p.Cols; j++ {
					p.Grad[i*p.Cols+j] += out.Grad[i*total+colOff+j]
				}
			}
			colOff += p.Cols
		}
	}
	return out
}

// Embedding gathers rows of `table` ([V,C]) at the given ids, producing [T,C].
func Embedding(table *Tensor, ids []int) *Tensor {
	C := table.Cols
	out := NewTensor(len(ids), C)
	for i, id := range ids {
		copy(out.Data[i*C:(i+1)*C], table.Data[id*C:(id+1)*C])
	}
	out.parents = []*Tensor{table}
	out.backward = func() {
		for i, id := range ids {
			for j := 0; j < C; j++ {
				table.Grad[id*C+j] += out.Grad[i*C+j]
			}
		}
	}
	return out
}

// CrossEntropy is the training loss: how surprised the model is by the correct
// next character at each position, averaged. logits is [T,V], targets is length
// T. Backward is the clean (softmax - onehot)/T.
func CrossEntropy(logits *Tensor, targets []int) *Tensor {
	T, V := logits.Rows, logits.Cols
	probs := make([]float64, T*V)
	var loss float64
	for i := 0; i < T; i++ {
		off := i * V
		max := math.Inf(-1)
		for j := 0; j < V; j++ {
			if logits.Data[off+j] > max {
				max = logits.Data[off+j]
			}
		}
		var sum float64
		for j := 0; j < V; j++ {
			e := math.Exp(logits.Data[off+j] - max)
			probs[off+j] = e
			sum += e
		}
		for j := 0; j < V; j++ {
			probs[off+j] /= sum
		}
		loss += -math.Log(probs[off+targets[i]] + 1e-12)
	}
	loss /= float64(T)

	out := NewTensor(1, 1)
	out.Data[0] = loss
	out.parents = []*Tensor{logits}
	out.backward = func() {
		scale := out.Grad[0] / float64(T)
		for i := 0; i < T; i++ {
			off := i * V
			for j := 0; j < V; j++ {
				g := probs[off+j]
				if j == targets[i] {
					g -= 1
				}
				logits.Grad[off+j] += g * scale
			}
		}
	}
	return out
}
