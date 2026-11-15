// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"strings"
	"testing"
)

func TestReversedRangeChangeIsNoop(t *testing.T) {
	f := newFixture(t, map[string]string{"p/p.go": "package p\n"}, fixtureOptions{})
	pURI := f.uri("p/p.go")
	f.open("p/p.go", 1)
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument": map[string]any{"uri": pURI, "version": 2},
		"contentChanges": []any{map[string]any{
			"range": map[string]any{
				"start": map[string]any{"line": 0, "character": 9},
				"end":   map[string]any{"line": 0, "character": 1},
			},
			"text": "INJECTED",
		}},
	})
	f.client.call("textDocument/hover", positionParams(pURI, Position{}))
	last := f.comp.bufferCalls()[len(f.comp.bufferCalls())-1]
	for path, text := range last {
		if strings.HasSuffix(path, "p.go") && string(text) != "package p\n" {
			t.Fatalf("a reversed range rewrote the document to %q", text)
		}
	}
}
