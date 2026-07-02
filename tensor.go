package tiny

// Tensor is one node in the computation graph. It holds a 2D matrix of values
// (Data) and the gradient of the final loss with respect to each value (Grad).
// This is "micrograd, but the value is a matrix": every operation records a
// backward closure that pushes gradient from this node to the nodes it was
// built from. Backward() runs them in reverse.
type Tensor struct {
	Data []float64 // row-major, length Rows*Cols
	Grad []float64 // same shape as Data
	Rows int
	Cols int

	backward func()    // pushes this node's Grad into its parents' Grad
	parents  []*Tensor // nodes this one was computed from
}

func NewTensor(rows, cols int) *Tensor {
	return &Tensor{
		Data: make([]float64, rows*cols),
		Grad: make([]float64, rows*cols),
		Rows: rows, Cols: cols,
	}
}

// Const wraps existing data as a leaf node.
func Const(rows, cols int, data []float64) *Tensor {
	t := NewTensor(rows, cols)
	copy(t.Data, data)
	return t
}

func (t *Tensor) At(r, c int) float64 { return t.Data[r*t.Cols+c] }

// MatMul returns a @ b. a is [m,k], b is [k,n], result is [m,n].
func MatMul(a, b *Tensor) *Tensor {
	if a.Cols != b.Rows {
		panic("matmul shape mismatch")
	}
	m, k, n := a.Rows, a.Cols, b.Cols
	out := NewTensor(m, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float64
			for p := 0; p < k; p++ {
				sum += a.Data[i*k+p] * b.Data[p*n+j]
			}
			out.Data[i*n+j] = sum
		}
	}
	out.parents = []*Tensor{a, b}
	out.backward = func() {
		// dA += dOut @ B^T ; dB += A^T @ dOut
		for i := 0; i < m; i++ {
			for p := 0; p < k; p++ {
				var g float64
				for j := 0; j < n; j++ {
					g += out.Grad[i*n+j] * b.Data[p*n+j]
				}
				a.Grad[i*k+p] += g
			}
		}
		for p := 0; p < k; p++ {
			for j := 0; j < n; j++ {
				var g float64
				for i := 0; i < m; i++ {
					g += a.Data[i*k+p] * out.Grad[i*n+j]
				}
				b.Grad[p*n+j] += g
			}
		}
	}
	return out
}

// Add is elementwise addition of two same-shape tensors.
func Add(a, b *Tensor) *Tensor {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("add shape mismatch")
	}
	out := NewTensor(a.Rows, a.Cols)
	for i := range a.Data {
		out.Data[i] = a.Data[i] + b.Data[i]
	}
	out.parents = []*Tensor{a, b}
	out.backward = func() {
		for i := range out.Grad {
			a.Grad[i] += out.Grad[i]
			b.Grad[i] += out.Grad[i]
		}
	}
	return out
}

// MulScalar multiplies every entry by s.
func MulScalar(a *Tensor, s float64) *Tensor {
	out := NewTensor(a.Rows, a.Cols)
	for i := range a.Data {
		out.Data[i] = a.Data[i] * s
	}
	out.parents = []*Tensor{a}
	out.backward = func() {
		for i := range out.Grad {
			a.Grad[i] += out.Grad[i] * s
		}
	}
	return out
}

// Backward computes gradients of this (scalar) node w.r.t. every node behind it.
func (t *Tensor) Backward() {
	// Build reverse topological order.
	var order []*Tensor
	seen := map[*Tensor]bool{}
	var visit func(n *Tensor)
	visit = func(n *Tensor) {
		if seen[n] {
			return
		}
		seen[n] = true
		for _, p := range n.parents {
			visit(p)
		}
		order = append(order, n)
	}
	visit(t)

	// Seed: d(loss)/d(loss) = 1.
	for i := range t.Grad {
		t.Grad[i] = 1
	}
	for i := len(order) - 1; i >= 0; i-- {
		if order[i].backward != nil {
			order[i].backward()
		}
	}
}

// ZeroGrad clears gradients across the whole graph behind t.
func (t *Tensor) ZeroGrad() {
	seen := map[*Tensor]bool{}
	var visit func(n *Tensor)
	visit = func(n *Tensor) {
		if seen[n] {
			return
		}
		seen[n] = true
		for i := range n.Grad {
			n.Grad[i] = 0
		}
		for _, p := range n.parents {
			visit(p)
		}
	}
	visit(t)
}
