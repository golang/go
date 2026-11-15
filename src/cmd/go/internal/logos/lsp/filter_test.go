// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"encoding/json"
	"strings"
	"testing"

	"cmd/internal/logos/editmap"
)

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

func TestFiltersKeepGoDocumentsWhole(t *testing.T) {
	pend := pending{method: "textDocument/completion"}
	in := []any{map[string]any{"label": "_logos_x"}}
	if got := prepareResult(pend, in); len(got.([]any)) != 1 {
		t.Errorf("a Go document's completion was filtered: %v", got)
	}
}

func TestInsideGeneratedBoundaryRuns(t *testing.T) {
	prefix := mustPosMap(t, "ab\n", "PRE\nab\n", editmap.EditMap{
		Version: editmap.Version, Source: "p.lgo", Size: 3, Out: 7,
		Edits:    []editmap.EditSpan{{Start: 0, End: 0, Len: 4}},
		Segments: spanSegs(3, []editmap.EditSpan{{Start: 0, End: 0, Len: 4}}),
	})
	for off, want := range map[int]bool{0: false, 1: true, 3: true, 4: false, 6: false, 7: false} {
		if got := prefix.insideGenerated(off); got != want {
			t.Errorf("prefix offset %d: insideGenerated = %v, want %v", off, got, want)
		}
	}

	edits := []editmap.EditSpan{{Start: 1, End: 1, Len: 3}}
	insert := mustPosMap(t, "ab", "aXYZb", editmap.EditMap{
		Version: editmap.Version, Source: "m.lgo", Size: 2, Out: 5,
		Edits: edits, Segments: spanSegs(2, edits),
	})
	for off, want := range map[int]bool{0: false, 1: false, 2: true, 3: true, 4: false, 5: false} {
		if got := insert.insideGenerated(off); got != want {
			t.Errorf("insertion offset %d: insideGenerated = %v, want %v", off, got, want)
		}
	}

	trailer := mustPosMap(t, "ab\n", "ab\nTRAILER\n", editmap.EditMap{
		Version: editmap.Version, Source: "t.lgo", Size: 3, Out: 11,
		Edits:    []editmap.EditSpan{{Start: 3, End: 3, Len: 8}},
		Segments: spanSegs(3, []editmap.EditSpan{{Start: 3, End: 3, Len: 8}}),
	})
	for off, want := range map[int]bool{0: false, 2: false, 3: false, 4: true, 10: true, 11: false} {
		if got := trailer.insideGenerated(off); got != want {
			t.Errorf("trailer offset %d: insideGenerated = %v, want %v", off, got, want)
		}
	}

	delEdits := []editmap.EditSpan{{Start: 4, End: 12, Len: 0}}
	deleted := mustPosMap(t, "one\ndropped\ntwo\n", "one\ntwo\n", editmap.EditMap{
		Version: editmap.Version, Source: "d.lgo", Size: 16, Out: 8,
		Edits: delEdits, Segments: spanSegs(16, delEdits),
	})
	for _, off := range []int{0, 3, 4, 7, 8} {
		if got := deleted.insideGenerated(off); got {
			t.Errorf("deletion offset %d counts as generated text", off)
		}
	}
}

func TestDropGeneratedHintsKeepsSeamHints(t *testing.T) {
	edits := []editmap.EditSpan{{Start: 1, End: 1, Len: 3}}
	pm := mustPosMap(t, "ab", "aXYZb", editmap.EditMap{
		Version: editmap.Version, Source: "m.lgo", Size: 2, Out: 5,
		Edits: edits, Segments: spanSegs(2, edits),
	})
	lw := &lowered{pm: pm}
	at := func(off int) map[string]any { return positionJSON(pm.low.position(off)) }
	in := []any{
		map[string]any{"position": at(0), "label": "copied-a"},
		map[string]any{"position": at(1), "label": "seam"},
		map[string]any{"position": at(2), "label": "inside"},
		map[string]any{"position": at(4), "label": "seam-after"},
		map[string]any{"position": at(5), "label": "copied-b"},
	}
	got, ok := dropGeneratedHints(in, lw).([]any)
	if !ok {
		t.Fatalf("dropGeneratedHints returned %T, want a list", dropGeneratedHints(in, lw))
	}
	kept := map[string]bool{}
	for _, item := range got {
		m, _ := item.(map[string]any)
		label, _ := m["label"].(string)
		kept[label] = true
	}
	for _, label := range []string{"copied-a", "seam", "seam-after", "copied-b"} {
		if !kept[label] {
			t.Errorf("hint %q dropped, want it kept", label)
		}
	}
	if kept["inside"] {
		t.Error("hint inside generated text kept, want it dropped")
	}
}
