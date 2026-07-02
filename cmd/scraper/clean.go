package main

import (
	"strings"

	tiny "github.com/erubboli/go-tiny-llm"
	"golang.org/x/net/html"
)

// extractText walks the HTML tree and concatenates visible text, skipping the
// contents of <script> and <style>.
func extractText(htmlSrc string) string {
	doc, err := html.Parse(strings.NewReader(htmlSrc))
	if err != nil {
		return ""
	}
	var sb strings.Builder
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode && (n.Data == "script" || n.Data == "style") {
			return
		}
		if n.Type == html.TextNode {
			sb.WriteString(n.Data)
			sb.WriteString(" ")
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(doc)
	return sb.String()
}

// cleanToVocab keeps only characters the model understands and collapses runs
// of whitespace into single spaces.
func cleanToVocab(s string) string {
	allowed := make(map[rune]bool)
	for _, r := range tiny.Vocab {
		allowed[r] = true
	}
	var sb strings.Builder
	lastSpace := false
	for _, r := range s {
		switch {
		case r == '\n' || r == '\t' || r == ' ':
			if !lastSpace {
				sb.WriteRune(' ')
				lastSpace = true
			}
		case allowed[r]:
			sb.WriteRune(r)
			lastSpace = false
		}
	}
	return strings.TrimSpace(sb.String())
}
