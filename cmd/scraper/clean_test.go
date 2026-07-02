package main

import (
	"strings"
	"testing"
)

func TestExtractTextDropsScript(t *testing.T) {
	html := `<html><body><p>Hello</p><script>var x=1;</script><p>World</p></body></html>`
	got := extractText(html)
	if strings.Contains(got, "var x") {
		t.Fatalf("script leaked into text: %q", got)
	}
	if !strings.Contains(got, "Hello") || !strings.Contains(got, "World") {
		t.Fatalf("visible text missing: %q", got)
	}
}

func TestCleanToVocabDropsUnknown(t *testing.T) {
	got := cleanToVocab("café — test")
	// 'é' and the em dash are not in Vocab; they vanish. 'caf' and 'test' stay.
	if strings.ContainsRune(got, 'é') || strings.Contains(got, "—") {
		t.Fatalf("unknown chars survived: %q", got)
	}
	if !strings.Contains(got, "caf") || !strings.Contains(got, "test") {
		t.Fatalf("valid text lost: %q", got)
	}
}
