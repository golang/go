// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"testing"

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

go 1.12
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

go 1.12
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
			"Foo",    // prefer exact segment first
			"Fooex",  // shorter than Fooest, FooBar, lexically befor Fooey
			"Fooey",  // shorter than Fooest, Foobar
			"FooBar", // TODO: prefer exact word matches.
			"Fooest",
		}
		syms := env.WorkspaceSymbol("Foo")
		if len(syms) != len(want) {
			t.Errorf("got %d symbols, want %d", len(syms), len(want))
		}

		for i := range syms {
			if syms[i].Name != want[i] {
				t.Errorf("syms[%d] = %q, want %q", i, syms[i].Name, want[i])
			}
		}
	})
}
