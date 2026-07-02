// Command scraper downloads one or more URLs, strips HTML, keeps only the
// model's vocabulary, and writes the result to data/input.txt. Not featured in
// the articles — the repo ships with data/input.txt already, so this is optional.
//
// Usage: go run ./cmd/scraper https://example.com/a https://example.com/b
package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

func main() {
	urls := os.Args[1:]
	if len(urls) == 0 {
		fmt.Println("usage: go run ./cmd/scraper <url> [url...]")
		os.Exit(1)
	}
	var all strings.Builder
	for _, u := range urls {
		resp, err := http.Get(u)
		if err != nil {
			fmt.Printf("skip %s: %v\n", u, err)
			continue
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		all.WriteString(cleanToVocab(extractText(string(body))))
		all.WriteString("\n")
		fmt.Printf("fetched %s (%d bytes raw)\n", u, len(body))
	}
	if err := os.WriteFile("data/input.txt", []byte(all.String()), 0644); err != nil {
		fmt.Println("write:", err)
		os.Exit(1)
	}
	fmt.Printf("wrote data/input.txt (%d chars)\n", all.Len())
}
