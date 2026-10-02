// Command stage6_distill demonstrates knowledge distillation end to end:
// a teacher (the stage5 shape) learns the book, then two identical small
// students train on the same batches, one from the book alone and one from a
// blend of the book and the teacher's softened opinions. The held-out loss
// shows what the teacher's opinion was worth. Run:
//
//	go run ./cmd/stage6_distill
package main

import (
	"fmt"
	"math/rand"
	"os"

	tiny "github.com/erubboli/go-tiny-llm"
)

// The student is deliberately much smaller than the teacher: a quarter of the
// embedding width, one layer instead of two, two heads instead of four. Its
// context window matches the teacher's so both can read the same batches.
var studentCfg = tiny.ModelConfig{
	EmbedDim:      16,
	NumLayers:     1,
	NumHeads:      2,
	ContextWindow: tiny.ContextWindow,
}

const (
	temperature = 4.0 // softens the teacher before the student imitates it
	alpha       = 0.5 // half hard loss (book), half soft loss (teacher)
	steps       = 3000
	evalWindows = 32
)

func main() {
	text, err := os.ReadFile("data/input.txt")
	if err != nil {
		fmt.Println("read data/input.txt:", err)
		os.Exit(1)
	}
	tok := tiny.NewTokenizer()
	ids := tok.Encode(string(text))
	split := len(ids) * 95 / 100 // hold out the last 5% for evaluation
	train, test := ids[:split], ids[split:]
	if len(train) < tiny.ContextWindow+2 || len(test) < tiny.ContextWindow+2 {
		fmt.Println("training text too short")
		os.Exit(1)
	}

	fmt.Printf("teacher:  embed %d, layers %d, heads %d, %d params\n",
		tiny.EmbedDim, tiny.NumLayers, tiny.NumHeads, paramCount(tiny.NewModel(tok.Size(), rand.New(rand.NewSource(0)))))
	fmt.Printf("student:  embed %d, layers %d, heads %d, %d params\n",
		studentCfg.EmbedDim, studentCfg.NumLayers, studentCfg.NumHeads,
		paramCount(tiny.NewModelConfig(tok.Size(), studentCfg, rand.New(rand.NewSource(0)))))

	// The teacher learns the book first. Same budget as stage5.
	teacherRng := rand.New(rand.NewSource(tiny.RandSeed))
	teacher := tiny.NewModel(tok.Size(), teacherRng)
	optT := tiny.NewAdamW(teacher.Params, tiny.LearningRate)
	fmt.Printf("\ntraining teacher (%d steps)\n", steps)
	for step := 0; step < steps; step++ {
		i := teacherRng.Intn(len(train) - tiny.ContextWindow)
		loss := teacher.Loss(train[i:i+tiny.ContextWindow], train[i+1:i+tiny.ContextWindow+1])
		optT.ZeroGrad()
		loss.Backward()
		optT.Step()
		if step%500 == 0 {
			fmt.Printf("  teacher step %5d/%d  loss %.4f\n", step, steps, loss.Data[0])
		}
	}

	// Two students, same shape and same seed, so only the loss differs.
	control := tiny.NewModelConfig(tok.Size(), studentCfg, rand.New(rand.NewSource(42)))
	distilled := tiny.NewModelConfig(tok.Size(), studentCfg, rand.New(rand.NewSource(42)))
	optC := tiny.NewAdamW(control.Params, tiny.LearningRate)
	optD := tiny.NewAdamW(distilled.Params, tiny.LearningRate)
	batchRng := rand.New(rand.NewSource(7))

	fmt.Printf("\ntraining students (%d steps, identical batches)\n", steps)
	for step := 0; step < steps; step++ {
		i := batchRng.Intn(len(train) - tiny.ContextWindow)
		in := train[i : i+tiny.ContextWindow]
		tgt := train[i+1 : i+tiny.ContextWindow+1]

		lossC := control.Loss(in, tgt)
		optC.ZeroGrad()
		lossC.Backward()
		optC.Step()

		// The frozen teacher donates logits; Detach stops any gradient from
		// flowing back into it.
		teacherLogits := tiny.Detach(teacher.Forward(in))
		lossD := tiny.DistillLoss(distilled.Forward(in), teacherLogits, tgt, temperature, alpha)
		optD.ZeroGrad()
		lossD.Backward()
		optD.Step()

		if step%500 == 0 {
			fmt.Printf("  step %5d/%d  control %.4f  distilled %.4f\n",
				step, steps, lossC.Data[0], lossD.Data[0])
		}
	}

	evalRng := rand.New(rand.NewSource(11))
	fmt.Println("\nheld-out cross-entropy (nats per character, lower is better):")
	fmt.Printf("  teacher (embed %d):   %.4f\n", tiny.EmbedDim, evaluate(teacher, evalRng, test))
	fmt.Printf("  control student:     %.4f\n", evaluate(control, evalRng, test))
	fmt.Printf("  distilled student:   %.4f\n", evaluate(distilled, evalRng, test))

	seed := tok.Encode("the queen said ")
	fmt.Println("\nsample from the control student:")
	fmt.Printf("  %s\n", string(tok.Decode(control.Generate(rand.New(rand.NewSource(5)), seed, 160))))
	fmt.Println("sample from the distilled student:")
	fmt.Printf("  %s\n", string(tok.Decode(distilled.Generate(rand.New(rand.NewSource(5)), seed, 160))))
}

// evaluate averages the next-character cross-entropy over fixed windows of
// the held-out data, so the three models are scored on identical text.
func evaluate(m *tiny.Model, rng *rand.Rand, ids []int) float64 {
	var sum float64
	for w := 0; w < evalWindows; w++ {
		i := rng.Intn(len(ids) - tiny.ContextWindow - 1)
		loss := m.Loss(ids[i:i+tiny.ContextWindow], ids[i+1:i+tiny.ContextWindow+1])
		sum += loss.Data[0]
	}
	return sum / evalWindows
}

func paramCount(m *tiny.Model) int {
	n := 0
	for _, p := range m.Params {
		n += len(p.Data)
	}
	return n
}
