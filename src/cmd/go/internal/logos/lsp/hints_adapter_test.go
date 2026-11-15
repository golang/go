// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"encoding/json"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"cmd/go/internal/logos/hints"
)

const wiringLgo = `package a

type User struct{ Name string }

func (u *User) Greet() string { return u.Name }

func name(u *User?) string {
	if u != nil {
		return u.Greet()
	}
	return ""
}

func get() (int, error) { return 1, nil }

func run() error {
	n := get()?
	_ = n
	return nil
}
`

type fixDiag struct {
	Schema, File             string
	Start                    Position
	Len                      int
	Description, Replacement string
}

func autofixLine(t *testing.T, d fixDiag) json.RawMessage {
	t.Helper()
	end := shift(d.Start, d.Len)
	return toJSON(t, map[string]any{
		"schema_version": d.Schema,
		"severity":       "deny",
		"rule_id":        "correctness::nil_deref_unproven",
		"file":           d.File,
		"span": map[string]int{
			"start_line": d.Start.Line + 1, "start_col": d.Start.Character + 1,
			"end_line": end.Line + 1, "end_col": end.Character + 1,
		},
		"message": "possible nil dereference of u",
		"autofix_candidates": []any{map[string]any{
			"description": d.Description,
			"replacement": d.Replacement,
		}},
	})
}

func TestProxyHintsEndToEnd(t *testing.T) {
	narrowed := shift(findPos(t, wiringLgo, "u.Greet", 0), len("u.Greet"))
	q := findPos(t, wiringLgo, "get()?", 0)
	site := span(q, len("get()?"))
	lowered := "n, _e := get()\n_logosTrace(_e)\nif _e != nil { return \"\", _e }"
	report := toJSON(t, map[string]any{
		"schemaVersion": "1.0",
		"package":       "example.com/w/a",
		"files": []any{map[string]any{
			"file": "a.lgo",
			"narrowed": []any{map[string]any{
				"position": map[string]any{"line": narrowed.Line, "character": narrowed.Character},
				"name":     "u",
				"type":     "*User",
			}},
			"unions": []any{},
			"sites": []any{map[string]any{
				"range": map[string]any{
					"start": map[string]any{"line": site.Start.Line, "character": site.Start.Character},
					"end":   map[string]any{"line": site.End.Line, "character": site.End.Character},
				},
				"kind":    "try",
				"lowered": lowered,
			}},
			"allocs": []any{},
		}},
	})
	f := newFixture(t, map[string]string{"a/a.lgo": wiringLgo}, fixtureOptions{
		Script: func(string) fakeScript {
			return fakeScript{Responses: map[string]json.RawMessage{
				"textDocument/inlayHint": toJSON(t, []any{}),
				"textDocument/hover": toJSON(t, map[string]any{
					"contents": map[string]any{"kind": "markdown", "value": "gopls hover"},
				}),
				"textDocument/codeAction": toJSON(t, []any{map[string]any{"title": "from gopls"}}),
			}}
		},
		Tune: func(f *fixture, o *Options) {
			o.HintsJSON = func(context.Context, string, map[string][]byte) (json.RawMessage, error) {
				return report, nil
			}
			o.Hints = hints.Serve
			o.CodeActions = codeActionsAdapter
		},
	})
	aPath := f.path("a/a.lgo")
	greet := findPos(t, wiringLgo, "u.Greet()", 0)
	var diags []Diagnostic
	for _, line := range []json.RawMessage{
		autofixLine(t, fixDiag{Schema: "2.0", File: aPath, Start: greet, Len: len("u.Greet()"), Description: "from another schema", Replacement: "x"}),
		autofixLine(t, fixDiag{Schema: "1.0", File: aPath, Start: greet, Len: len("u.Greet()"), Description: "use optional chaining with a ?? default", Replacement: `(u?.Greet() ?? "")`}),
	} {
		d, ok := parseDiagnostic(line)
		if !ok {
			t.Fatalf("parseDiagnostic rejected %s", line)
		}
		diags = append(diags, d)
	}
	f.comp.setDiagnostics(filepath.Dir(aPath), diags...)
	f.open("a/a.lgo", 1)

	inlay := f.client.call("textDocument/inlayHint", map[string]any{
		"textDocument": map[string]any{"uri": f.uri("a/a.lgo")},
		"range": map[string]any{
			"start": map[string]any{"line": 0, "character": 0},
			"end":   map[string]any{"line": 1000, "character": 0},
		},
	})
	if !strings.Contains(string(inlay), "narrowed:") || !strings.Contains(string(inlay), "*User") {
		t.Errorf("inlayHint lacks the narrowed hint: %s", inlay)
	}

	hover := f.client.call("textDocument/hover", map[string]any{
		"textDocument": map[string]any{"uri": f.uri("a/a.lgo")},
		"position":     q,
	})
	var hoverBody struct {
		Contents struct {
			Value string `json:"value"`
		} `json:"contents"`
	}
	if err := json.Unmarshal(hover, &hoverBody); err != nil {
		t.Fatalf("hover response is not JSON: %v\n%s", err, hover)
	}
	if !strings.Contains(hoverBody.Contents.Value, lowered) {
		t.Errorf("hover lacks the reported lowered code %q:\n%s", lowered, hoverBody.Contents.Value)
	}
	if !strings.Contains(hoverBody.Contents.Value, "gopls hover") {
		t.Errorf("hover lost the gopls text: %s", hoverBody.Contents.Value)
	}

	actions := f.client.call("textDocument/codeAction", map[string]any{
		"textDocument": map[string]any{"uri": f.uri("a/a.lgo")},
		"range":        span(greet, len("u.Greet()")),
		"context":      map[string]any{"diagnostics": []any{}},
	})
	if !strings.Contains(string(actions), "from gopls") {
		t.Errorf("codeAction lost the gopls action: %s", actions)
	}
	if !strings.Contains(string(actions), "use optional chaining with a ?? default") {
		t.Errorf("codeAction lacks the autofix candidate: %s", actions)
	}
	if strings.Contains(string(actions), "from another schema") {
		t.Errorf("codeAction offers the candidate of a diagnostic of another schema: %s", actions)
	}
}

const adapterSrc = "package p\n\nfunc name(u *User?) string {\n\treturn u.Name\n}\n"

func nameSpan(t *testing.T) (Position, json.RawMessage) {
	t.Helper()
	off := strings.Index(adapterSrc, "u.Name")
	start := newText([]byte(adapterSrc)).position(off)
	return start, toJSON(t, span(start, len("u.Name")))
}

func titles(t *testing.T, raw json.RawMessage) []string {
	t.Helper()
	var acts []hints.CodeAction
	if err := json.Unmarshal(raw, &acts); err != nil {
		t.Fatalf("adapter result %s is not an array of code actions: %v", raw, err)
	}
	var out []string
	for _, a := range acts {
		out = append(out, a.Title)
	}
	return out
}

func TestCodeActionsAdapterAppliesAutofix(t *testing.T) {
	start, rng := nameSpan(t)
	line := autofixLine(t, fixDiag{Schema: "1.0", File: "/w/fixes.lgo", Start: start, Len: len("u.Name"), Description: "use optional chaining with a ?? default", Replacement: `(u?.Name ?? "")`})
	raw, err := codeActionsAdapter([][]byte{line}, "file:///w/fixes.lgo", []byte(adapterSrc), rng)
	if err != nil {
		t.Fatal(err)
	}
	var acts []hints.CodeAction
	if err := json.Unmarshal(raw, &acts); err != nil || len(acts) != 1 {
		t.Fatalf("adapter result = %s, err = %v, want one action", raw, err)
	}
	a := acts[0]
	if a.Kind != "quickfix" || a.Title != "use optional chaining with a ?? default" {
		t.Errorf("action = %s %q, want the quickfix of the candidate", a.Kind, a.Title)
	}
	edits := a.Edit.Changes["file:///w/fixes.lgo"]
	want := hints.TextEdit{
		Range:   hints.Range{Start: hints.Position(start), End: hints.Position(shift(start, len("u.Name")))},
		NewText: `(u?.Name ?? "")`,
	}
	if len(edits) != 1 || edits[0] != want {
		t.Errorf("edits = %+v, want %+v", edits, want)
	}
}

func TestCodeActionsAdapterSkipsBadLines(t *testing.T) {
	start, rng := nameSpan(t)
	const uri = "file:///w/fixes.lgo"
	line := func(schema, description string) []byte {
		return autofixLine(t, fixDiag{Schema: schema, File: "/w/fixes.lgo", Start: start, Len: len("u.Name"), Description: description, Replacement: "x"})
	}
	lines := [][]byte{
		[]byte("go: finding module for package"),
		line("1.0", "first fix"),
		line("2.0", "another schema"),
		[]byte(`{"schema_version":"1.0","span":"not a span"}`),
		line("1.0", "second fix"),
		nil,
	}
	raw, err := codeActionsAdapter(lines, uri, []byte(adapterSrc), rng)
	if err != nil {
		t.Fatalf("a bad diagnostic line made the adapter fail: %v", err)
	}
	if got, want := titles(t, raw), []string{"first fix", "second fix"}; !slices.Equal(got, want) {
		t.Errorf("titles = %q, want %q", got, want)
	}
}

func TestCodeActionsAdapterEmptyIsArray(t *testing.T) {
	start, rng := nameSpan(t)
	good := autofixLine(t, fixDiag{Schema: "1.0", File: "/w/fixes.lgo", Start: start, Len: len("u.Name"), Description: "fix", Replacement: "x"})
	elsewhere := toJSON(t, Range{Start: Position{Line: 0, Character: 0}, End: Position{Line: 0, Character: 3}})
	for _, tc := range []struct {
		name  string
		diags [][]byte
		rng   json.RawMessage
	}{
		{"no diagnostics", nil, rng},
		{"only unreadable lines", [][]byte{[]byte("not json"), toJSON(t, map[string]any{"schema_version": "2.0"})}, rng},
		{"range off the diagnostic", [][]byte{good}, elsewhere},
	} {
		raw, err := codeActionsAdapter(tc.diags, "file:///w/fixes.lgo", []byte(adapterSrc), tc.rng)
		if err != nil {
			t.Errorf("%s: %v", tc.name, err)
			continue
		}
		if string(raw) != "[]" {
			t.Errorf("%s: adapter result = %s, want []", tc.name, raw)
		}
	}
}

func TestCodeActionsAdapterRejectsBadRange(t *testing.T) {
	if raw, err := codeActionsAdapter(nil, "file:///w/fixes.lgo", []byte(adapterSrc), json.RawMessage(`{"start":"here"}`)); err == nil {
		t.Errorf("adapter result = %s, want an error for an unreadable range", raw)
	}
}
