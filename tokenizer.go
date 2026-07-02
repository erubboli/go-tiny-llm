package tiny

// Tokenizer maps characters to integer ids and back. Character-level: no BPE,
// no subwords. The whole vocabulary is the Vocab constant.
type Tokenizer struct {
	runes  []rune       // id -> char
	lookup map[rune]int // char -> id
}

func NewTokenizer() *Tokenizer {
	runes := []rune(Vocab)
	lookup := make(map[rune]int, len(runes))
	for i, r := range runes {
		lookup[r] = i
	}
	return &Tokenizer{runes: runes, lookup: lookup}
}

func (t *Tokenizer) Size() int { return len(t.runes) }

// Encode turns text into ids. Characters not in the vocabulary are skipped.
func (t *Tokenizer) Encode(s string) []int {
	out := make([]int, 0, len(s))
	for _, r := range s {
		if id, ok := t.lookup[r]; ok {
			out = append(out, id)
		}
	}
	return out
}

func (t *Tokenizer) Decode(ids []int) string {
	out := make([]rune, len(ids))
	for i, id := range ids {
		out[i] = t.runes[id]
	}
	return string(out)
}
