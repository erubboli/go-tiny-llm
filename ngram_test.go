package tiny

import "testing"

func TestNGramTableSizeExplodes(t *testing.T) {
	if got := NGramTableSize(70, 1); got != 70 {
		t.Fatalf("1-gram: got %v want 70", got)
	}
	if got := NGramTableSize(70, 2); got != 4900 {
		t.Fatalf("2-gram: got %v want 4900", got)
	}
	// A 10-character context over 70 symbols needs > 1e18 rows.
	if NGramTableSize(70, 10) < 1e18 {
		t.Fatalf("10-gram should exceed 1e18, got %v", NGramTableSize(70, 10))
	}
}
