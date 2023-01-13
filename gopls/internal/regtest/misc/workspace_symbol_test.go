// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	. "golang.org/x/tools/gopls/internal/lsp/regtest"
	"golang.org/x/tools/gopls/internal/lsp/source"
)

func TestWorkspaceSymbolMissingMetadata(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.17
-- a.go --
package p

const C1 = "a.go"
-- exclude.go --

//go:build exclude
// +build exclude

package exclude

const C2 = "exclude.go"
`

	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("a.go")
		syms := env.Symbol("C")
		if got, want := len(syms), 1; got != want {
			t.Errorf("got %d symbols, want %d", got, want)
		}

		// Opening up an ignored file will result in an overlay with missing
		// metadata, but this shouldn't break workspace symbols requests.
		env.OpenFile("exclude.go")
		syms = env.Symbol("C")
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
		Settings{"symbolMatcher": symbolMatcher},
	).Run(t, files, func(t *testing.T, env *Env) {
		want := []string{
			"Foo",    // prefer exact segment matches first
			"FooBar", // ...followed by exact word matches
			"Fooex",  // shorter than Fooest, FooBar, lexically before Fooey
			"Fooey",  // shorter than Fooest, Foobar
			"Fooest",
		}
		got := env.Symbol("Foo")
		compareSymbols(t, got, want...)
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
		Settings{"symbolMatcher": symbolMatcher},
	).Run(t, files, func(t *testing.T, env *Env) {
		compareSymbols(t, env.Symbol("ABC"), "ABC", "AxxBxxCxx")
		compareSymbols(t, env.Symbol("'ABC"), "ABC")
		compareSymbols(t, env.Symbol("^mod.com"), "mod.com/a.ABC", "mod.com/a.AxxBxxCxx")
		compareSymbols(t, env.Symbol("^mod.com Axx"), "mod.com/a.AxxBxxCxx")
		compareSymbols(t, env.Symbol("C$"), "ABC")
	})
}

func compareSymbols(t *testing.T, got []protocol.SymbolInformation, want ...string) {
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
