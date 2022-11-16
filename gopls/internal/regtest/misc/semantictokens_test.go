// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/tools/gopls/internal/lsp"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	. "golang.org/x/tools/gopls/internal/lsp/regtest"
	"golang.org/x/tools/internal/typeparams"
)

func TestBadURICrash_VSCodeIssue1498(t *testing.T) {
	const src = `
-- go.mod --
module example.com

go 1.12

-- main.go --
package main

func main() {}

`
	WithOptions(
		Modes(Default),
		Settings{"allExperiments": true},
	).Run(t, src, func(t *testing.T, env *Env) {
		params := &protocol.SemanticTokensParams{}
		const badURI = "http://foo"
		params.TextDocument.URI = badURI
		// This call panicked in the past: golang/vscode-go#1498.
		if _, err := env.Editor.Server.SemanticTokensFull(env.Ctx, params); err != nil {
			// Requests to an invalid URI scheme shouldn't result in an error, we
			// simply don't support this so return empty result. This could be
			// changed, but for now assert on the current behavior.
			t.Errorf("SemanticTokensFull(%q): %v", badURI, err)
		}
	})
}

// fix bug involving type parameters and regular parameters
// (golang/vscode-go#2527)
func TestSemantic_2527(t *testing.T) {
	if !typeparams.Enabled {
		t.Skip("type parameters are needed for this test")
	}
	// these are the expected types of identfiers in textt order
	want := []result{
		{"package", "keyword", ""},
		{"foo", "namespace", ""},
		{"func", "keyword", ""},
		{"Add", "function", "definition deprecated"},
		{"T", "typeParameter", "definition"},
		{"int", "type", "defaultLibrary"},
		{"target", "parameter", "definition"},
		{"T", "typeParameter", ""},
		{"l", "parameter", "definition"},
		{"T", "typeParameter", ""},
		{"T", "typeParameter", ""},
		{"return", "keyword", ""},
		{"append", "function", "defaultLibrary"},
		{"l", "parameter", ""},
		{"target", "parameter", ""},
		{"for", "keyword", ""},
		{"range", "keyword", ""},
		{"l", "parameter", ""},
		{"return", "keyword", ""},
		{"nil", "variable", "readonly defaultLibrary"},
	}
	src := `
-- go.mod --
module example.com

go 1.19
-- main.go --
package foo
// Deprecated (for testing)
func Add[T int](target T, l []T) []T {
	return append(l, target)
	for range l {} // test coverage
	return nil
}
`
	WithOptions(
		Modes(Default),
		Settings{"semanticTokens": true},
	).Run(t, src, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.AfterChange(env.DiagnosticAtRegexp("main.go", "for range"))
		p := &protocol.SemanticTokensParams{
			TextDocument: protocol.TextDocumentIdentifier{
				URI: env.Sandbox.Workdir.URI("main.go"),
			},
		}
		v, err := env.Editor.Server.SemanticTokensFull(env.Ctx, p)
		if err != nil {
			t.Fatal(err)
		}
		seen := interpret(v.Data, env.Editor.BufferText("main.go"))
		if x := cmp.Diff(want, seen); x != "" {
			t.Errorf("Semantic tokens do not match (-want +got):\n%s", x)
		}
	})

}

type result struct {
	Token     string
	TokenType string
	Mod       string
}

// human-readable version of the semantic tokens
// comment, string, number are elided
// (and in the future, maybe elide other things, like operators)
func interpret(x []uint32, contents string) []result {
	lines := strings.Split(contents, "\n")
	ans := []result{}
	line, col := 1, 1
	for i := 0; i < len(x); i += 5 {
		line += int(x[i])
		col += int(x[i+1])
		if x[i] != 0 { // new line
			col = int(x[i+1]) + 1 // 1-based column numbers
		}
		sz := x[i+2]
		t := semanticTypes[x[i+3]]
		if t == "comment" || t == "string" || t == "number" {
			continue
		}
		l := x[i+4]
		var mods []string
		for i, mod := range semanticModifiers {
			if l&(1<<i) != 0 {
				mods = append(mods, mod)
			}
		}
		// col is a utf-8 offset
		tok := lines[line-1][col-1 : col-1+int(sz)]
		ans = append(ans, result{tok, t, strings.Join(mods, " ")})
	}
	return ans
}

var (
	semanticTypes     = lsp.SemanticTypes()
	semanticModifiers = lsp.SemanticModifiers()
)
