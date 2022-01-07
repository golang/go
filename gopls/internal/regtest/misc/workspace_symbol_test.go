// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"testing"

	"golang.org/x/tools/internal/lsp/protocol"
	. "golang.org/x/tools/internal/lsp/regtest"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/testenv"
)

func TestWorkspaceSymbolMissingMetadata(t *testing.T) {
	// We get 2 symbols on 1.12, for some reason.
	testenv.NeedsGo1Point(t, 13)

	const files = `
-- go.mod --
module mod.com

go 1.17
-- a.go --
package p

const C1 = "a.go"
-- ignore.go --

// +build ignore

package ignore

const C2 = "ignore.go"
`

	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("a.go")
		syms := env.WorkspaceSymbol("C")
		if got, want := len(syms), 1; got != want {
			t.Errorf("got %d symbols, want %d", got, want)
		}

		// Opening up an ignored file will result in an overlay with missing
		// metadata, but this shouldn't break workspace symbols requests.
		env.OpenFile("ignore.go")
		syms = env.WorkspaceSymbol("C")
		if got, want := len(syms), 1; got != want {
			t.Errorf("got %d symbols, want %d", got, want)
		}
	})
}

func TestWorkspaceSymbolSorting(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.17
-- a/a.go --
package a

const (
	Foo = iota
	FooBar
	Fooey
	Fooex
	Fooest
)
`

	var symbolMatcher = string(source.SymbolFastFuzzy)
	WithOptions(
		EditorConfig{
			SymbolMatcher: &symbolMatcher,
		},
	).Run(t, files, func(t *testing.T, env *Env) {
		want := []string{
			"Foo",    // prefer exact segment matches first
			"FooBar", // ...followed by exact word matches
			"Fooex",  // shorter than Fooest, FooBar, lexically before Fooey
			"Fooey",  // shorter than Fooest, Foobar
			"Fooest",
		}
		got := env.WorkspaceSymbol("Foo")
		compareSymbols(t, got, want)
	})
}

func TestWorkspaceSymbolSpecialPatterns(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.17
-- a/a.go --
package a

const (
	AxxBxxCxx
	ABC
)
`

	var symbolMatcher = string(source.SymbolFastFuzzy)
	WithOptions(
		EditorConfig{
			SymbolMatcher: &symbolMatcher,
		},
	).Run(t, files, func(t *testing.T, env *Env) {
		compareSymbols(t, env.WorkspaceSymbol("ABC"), []string{"ABC", "AxxBxxCxx"})
		compareSymbols(t, env.WorkspaceSymbol("'ABC"), []string{"ABC"})
		compareSymbols(t, env.WorkspaceSymbol("^mod.com"), []string{"mod.com/a.ABC", "mod.com/a.AxxBxxCxx"})
		compareSymbols(t, env.WorkspaceSymbol("^mod.com Axx"), []string{"mod.com/a.AxxBxxCxx"})
		compareSymbols(t, env.WorkspaceSymbol("C$"), []string{"ABC"})
	})
}

func compareSymbols(t *testing.T, got []protocol.SymbolInformation, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("got %d symbols, want %d", len(got), len(want))
	}

	for i := range got {
		if got[i].Name != want[i] {
			t.Errorf("got[%d] = %q, want %q", i, got[i].Name, want[i])
		}
	}
}
