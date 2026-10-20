// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestInsideGeneratedExcludesSeams checks the predicate behind the inlay
// hint filter: strictly inside generated text is inside, and the first and
// last offsets of it, the seams with the user's own text, are not.
func TestInsideGeneratedExcludesSeams(t *testing.T) {
	src, out, pm := questionMap(t)
	srcLine := lineOf(src, 5)
	dataEnd := pm.low.offset(Position{Line: 5, Character: strings.Index(srcLine, "data") + len("data")})
	cases := []struct {
		name string
		line int
		col  int
		want bool
	}{
		{"the end of data, where the user's own text meets the generated variable", 5, strings.Index(srcLine, "data") + len("data"), false},
		{"inside the generated error variable", 5, strings.Index(lineOf(out, 5), "_e") + 2, true},
		{"inside the generated trace line", 6, 3, true},
		{"the start of the generated line", 6, 0, true},
		{"the user's own text", 7 + 1, 4, false},
	}
	for _, c := range cases {
		if got := pm.insideGenerated(pm.low.offset(Position{Line: c.line, Character: c.col})); got != c.want {
			t.Errorf("%s: insideGenerated = %v, want %v", c.name, got, c.want)
		}
	}
	if pm.insideGenerated(dataEnd) {
		t.Error("the end of data counts as generated text")
	}
	if pm.insideGenerated(0) || pm.insideGenerated(len(out)) {
		t.Error("the ends of the text count as generated text")
	}
}

// TestProxyDropsWhatTheLoweringInvented checks the answer filters: an
// inlay hint inside the generated error variable of a ? statement is not
// shown, while the hint at the end of data, where the user's text meets
// the generated text, is; completions and symbols named with the reserved
// prefix are not shown, at any depth of the symbol tree.
func TestProxyDropsWhatTheLoweringInvented(t *testing.T) {
	low, _, _ := lowerQuestion([]byte(aLgo))
	lowText := string(low)
	lowData := shift(findPos(t, lowText, "data", 0), len("data"))
	lowErr := shift(findPos(t, lowText, "_e", 0), len("_e"))
	lowTrace := Position{Line: 6, Character: 3}
	origData := shift(findPos(t, aLgo, "data", 0), len("data"))
	whole := Range{Start: Position{}, End: Position{Line: 9}}

	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{
		Script: func(string) fakeScript {
			return fakeScript{Responses: map[string]json.RawMessage{
				"textDocument/inlayHint": toJSON(t, []any{
					map[string]any{"position": lowData, "label": ": []byte"},
					map[string]any{"position": lowErr, "label": ": error"},
					map[string]any{"position": lowTrace, "label": "generated"},
				}),
				"textDocument/completion": toJSON(t, map[string]any{
					"isIncomplete": false,
					"items": []any{
						map[string]any{"label": "_logos_rt"},
						map[string]any{"label": "ReadFile"},
						map[string]any{"label": "_logos_site0"},
					},
				}),
				"textDocument/documentSymbol": toJSON(t, []any{
					map[string]any{"name": "Read", "kind": 12, "range": whole, "selectionRange": whole, "children": []any{
						map[string]any{"name": "_logos_e0", "kind": 13, "range": whole, "selectionRange": whole},
						map[string]any{"name": "data", "kind": 13, "range": whole, "selectionRange": whole},
					}},
					map[string]any{"name": "_logos_site0", "kind": 13, "range": whole, "selectionRange": whole},
				}),
			}}
		},
	})
	f.open("a/a.lgo", 1)
	aURI := f.uri("a/a.lgo")

	res := f.client.call("textDocument/inlayHint", map[string]any{"textDocument": map[string]any{"uri": aURI}, "range": whole})
	sameJSON(t, "inlay hints", res, []any{map[string]any{"position": origData, "label": ": []byte"}})

	res = f.client.call("textDocument/completion", positionParams(aURI, Position{Line: 5, Character: 2}))
	sameJSON(t, "completion", res, map[string]any{
		"isIncomplete": false,
		"items":        []any{map[string]any{"label": "ReadFile"}},
	})

	res = f.client.call("textDocument/documentSymbol", map[string]any{"textDocument": map[string]any{"uri": aURI}})
	var symbols []struct {
		Name     string `json:"name"`
		Children []struct {
			Name string `json:"name"`
		} `json:"children"`
	}
	if err := json.Unmarshal(res, &symbols); err != nil {
		t.Fatal(err)
	}
	if len(symbols) != 1 || symbols[0].Name != "Read" || len(symbols[0].Children) != 1 || symbols[0].Children[0].Name != "data" {
		t.Errorf("document symbols = %s, want Read with the child data only", res)
	}
}

// TestFiltersKeepGoDocumentsWhole checks that a request on a Go document,
// which has no lowering, keeps every item of the answer.
func TestFiltersKeepGoDocumentsWhole(t *testing.T) {
	pend := pending{method: "textDocument/completion"}
	in := []any{map[string]any{"label": "_logos_x"}}
	if got := prepareResult(pend, in); len(got.([]any)) != 1 {
		t.Errorf("a Go document's completion was filtered: %v", got)
	}
}
