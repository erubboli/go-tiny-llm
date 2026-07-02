package tiny

import "testing"

func TestEncodeDecodeRoundTrip(t *testing.T) {
	tok := NewTokenizer()
	in := "Hello, world!"
	got := tok.Decode(tok.Encode(in))
	if got != in {
		t.Fatalf("round trip: got %q want %q", got, in)
	}
}

func TestEncodeSkipsUnknown(t *testing.T) {
	tok := NewTokenizer()
	// The tilde is not in Vocab and must be dropped.
	ids := tok.Encode("a~b")
	if got := tok.Decode(ids); got != "ab" {
		t.Fatalf("unknown handling: got %q want %q", got, "ab")
	}
}

func TestSizeMatchesVocab(t *testing.T) {
	tok := NewTokenizer()
	if tok.Size() != len([]rune(Vocab)) {
		t.Fatalf("size %d want %d", tok.Size(), len([]rune(Vocab)))
	}
}
