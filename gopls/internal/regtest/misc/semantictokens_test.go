// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/tools/gopls/internal/lsp/fake"
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
	// these are the expected types of identifiers in text order
	want := []fake.SemanticToken{
		{Token: "package", TokenType: "keyword"},
		{Token: "foo", TokenType: "namespace"},
		{Token: "// Deprecated (for testing)", TokenType: "comment"},
		{Token: "func", TokenType: "keyword"},
		{Token: "Add", TokenType: "function", Mod: "definition deprecated"},
		{Token: "T", TokenType: "typeParameter", Mod: "definition"},
		{Token: "int", TokenType: "type", Mod: "defaultLibrary"},
		{Token: "target", TokenType: "parameter", Mod: "definition"},
		{Token: "T", TokenType: "typeParameter"},
		{Token: "l", TokenType: "parameter", Mod: "definition"},
		{Token: "T", TokenType: "typeParameter"},
		{Token: "T", TokenType: "typeParameter"},
		{Token: "return", TokenType: "keyword"},
		{Token: "append", TokenType: "function", Mod: "defaultLibrary"},
		{Token: "l", TokenType: "parameter"},
		{Token: "target", TokenType: "parameter"},
		{Token: "for", TokenType: "keyword"},
		{Token: "range", TokenType: "keyword"},
		{Token: "l", TokenType: "parameter"},
		{Token: "// test coverage", TokenType: "comment"},
		{Token: "return", TokenType: "keyword"},
		{Token: "nil", TokenType: "variable", Mod: "readonly defaultLibrary"},
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
		env.AfterChange(
			Diagnostics(env.AtRegexp("main.go", "for range")),
		)
		seen, err := env.Editor.SemanticTokens(env.Ctx, "main.go")
		if err != nil {
			t.Fatal(err)
		}
		if x := cmp.Diff(want, seen); x != "" {
			t.Errorf("Semantic tokens do not match (-want +got):\n%s", x)
		}
	})

}

// fix inconsistency in TypeParameters
// https://github.com/golang/go/issues/57619
func TestSemantic_57619(t *testing.T) {
	if !typeparams.Enabled {
		t.Skip("type parameters are needed for this test")
	}
	src := `
-- go.mod --
module example.com

go 1.19
-- main.go --
package foo
type Smap[K int, V any] struct {
	Store map[K]V
}
func (s *Smap[K, V]) Get(k K) (V, bool) {
	v, ok := s.Store[k]
	return v, ok
}
func New[K int, V any]() Smap[K, V] {
	return Smap[K, V]{Store: make(map[K]V)}
}
`
	WithOptions(
		Modes(Default),
		Settings{"semanticTokens": true},
	).Run(t, src, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		seen, err := env.Editor.SemanticTokens(env.Ctx, "main.go")
		if err != nil {
			t.Fatal(err)
		}
		for i, s := range seen {
			if (s.Token == "K" || s.Token == "V") && s.TokenType != "typeParameter" {
				t.Errorf("%d: expected K and V to be type parameters, but got %v", i, s)
			}
		}
	})
}
