// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"testing"

	"github.com/google/go-cmp/cmp"
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

const K1 = "a.go"
-- exclude.go --

//go:build exclude
// +build exclude

package exclude

const K2 = "exclude.go"
`

	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("a.go")
		checkSymbols(env, "K", "K1")

		// Opening up an ignored file will result in an overlay with missing
		// metadata, but this shouldn't break workspace symbols requests.
		env.OpenFile("exclude.go")
		checkSymbols(env, "K", "K1")
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
		checkSymbols(env, "Foo",
			"Foo",    // prefer exact segment matches first
			"FooBar", // ...followed by exact word matches
			"Fooex",  // shorter than Fooest, FooBar, lexically before Fooey
			"Fooey",  // shorter than Fooest, Foobar
			"Fooest",
		)
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
		checkSymbols(env, "ABC", "ABC", "AxxBxxCxx")
		checkSymbols(env, "'ABC", "ABC")
		checkSymbols(env, "^mod.com", "mod.com/a.ABC", "mod.com/a.AxxBxxCxx")
		checkSymbols(env, "^mod.com Axx", "mod.com/a.AxxBxxCxx")
		checkSymbols(env, "C$", "ABC")
	})
}

func checkSymbols(env *Env, query string, want ...string) {
	env.T.Helper()
	var got []string
	for _, info := range env.Symbol(query) {
		got = append(got, info.Name)
	}
	if diff := cmp.Diff(got, want); diff != "" {
		env.T.Errorf("unexpected Symbol(%q) result (+want -got):\n%s", query, diff)
	}
}
