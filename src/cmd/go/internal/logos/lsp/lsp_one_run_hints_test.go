// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
)

func TestProxyOneRunSharedAcrossHoverHintsDiagnostics(t *testing.T) {
	var lowData Range
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{
		Script: func(root string) fakeScript {
			low, _, _ := lowerQuestion([]byte(aLgo))
			lowData = span(findPos(t, string(low), "data", 0), len("data"))
			return fakeScript{Responses: map[string]json.RawMessage{
				"textDocument/inlayHint": toJSON(t, []any{map[string]any{"position": lowData, "label": ": []byte"}}),
			}}
		},
	})
	aPath := f.path("a/a.lgo")
	aURI := f.uri("a/a.lgo")
	callLine, callCol := 6, strings.Index(lineOf(aLgo, 5), "os.ReadFile")+1
	f.comp.setDiagnostics(filepath.Dir(aPath),
		compilerDiag(t, diagSpec{Path: aPath, Severity: "deny", Rule: "correctness::example", Message: "compiler: unproven",
			Start: [2]int{callLine, callCol}, End: [2]int{callLine, callCol + len("os.ReadFile")}}),
	)
	f.open("a/a.lgo", 1)
	edited2 := aLgo + "// v2\n"
	edited3 := aLgo + "// v2\n// v3\n"
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": aURI, "version": 2},
		"contentChanges": []any{map[string]any{"text": edited2}},
	})
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": aURI, "version": 3},
		"contentChanges": []any{map[string]any{"text": edited3}},
	})
	operand := shift(findPos(t, edited3, "ReadFile(path)?", 0), len("ReadFile("))
	f.client.call("textDocument/hover", positionParams(aURI, operand))
	f.client.call("textDocument/inlayHint", map[string]any{
		"textDocument": map[string]any{"uri": aURI},
		"range":        Range{Start: Position{Line: 0}, End: Position{Line: 8}},
	})
	if got := len(f.comp.lowerCalls()); got != 1 {
		t.Fatalf("hover plus inlayHint compiled %d times, want the one shared run", got)
	}
	params := f.client.notification(func(m map[string]any) bool {
		items, _ := m["diagnostics"].([]any)
		if m["uri"] != aURI {
			return false
		}
		for _, item := range items {
			d, _ := item.(map[string]any)
			if d["message"] == "compiler: unproven" {
				return true
			}
		}
		return false
	})
	if v, _ := number(params["version"]); v != 3 {
		t.Errorf("diagnostics published for version %v, want the client version 3", params["version"])
	}
	if got := len(f.comp.lowerCalls()); got != 1 {
		t.Fatalf("diagnostics plus hints compiled %d times, want the one shared run", got)
	}
}
