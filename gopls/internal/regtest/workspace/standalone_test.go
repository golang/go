// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package workspace

import (
	"sort"
	"testing"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	. "golang.org/x/tools/gopls/internal/lsp/regtest"
)

func TestStandaloneFiles(t *testing.T) {
	const files = `
-- go.mod --
module mod.test

go 1.16
-- lib/lib.go --
package lib

const K = 0

type I interface {
	M()
}
-- lib/ignore.go --
//go:build ignore
// +build ignore

package main

import (
	"mod.test/lib"
)

const K = 1

type Mer struct{}
func (Mer) M()

func main() {
	println(lib.K + K)
}
`
	WithOptions(
		// On Go 1.17 and earlier, this test fails with
		// experimentalWorkspaceModule. Not investigated, as
		// experimentalWorkspaceModule will be removed.
		Modes(Default),
	).Run(t, files, func(t *testing.T, env *Env) {
		// Initially, gopls should not know about the standalone file as it hasn't
		// been opened. Therefore, we should only find one symbol 'K'.
		//
		// (The choice of "K" is a little sleazy: it was originally "C" until
		// we started adding "unsafe" to the workspace unconditionally, which
		// caused a spurious match of "unsafe.Slice". But in practice every
		// workspace depends on unsafe.)
		syms := env.Symbol("K")
		if got, want := len(syms), 1; got != want {
			t.Errorf("got %d symbols, want %d (%+v)", got, want, syms)
		}

		// Similarly, we should only find one reference to "K", and no
		// implementations of I.
		checkLocations := func(method string, gotLocations []protocol.Location, wantFiles ...string) {
			var gotFiles []string
			for _, l := range gotLocations {
				gotFiles = append(gotFiles, env.Sandbox.Workdir.URIToPath(l.URI))
			}
			sort.Strings(gotFiles)
			sort.Strings(wantFiles)
			if diff := cmp.Diff(wantFiles, gotFiles); diff != "" {
				t.Errorf("%s(...): unexpected locations (-want +got):\n%s", method, diff)
			}
		}

		env.OpenFile("lib/lib.go")
		env.AfterChange(NoDiagnostics())

		// Replacing K with D should not cause any workspace diagnostics, since we
		// haven't yet opened the standalone file.
		env.RegexpReplace("lib/lib.go", "K", "D")
		env.AfterChange(NoDiagnostics())
		env.RegexpReplace("lib/lib.go", "D", "K")
		env.AfterChange(NoDiagnostics())

		refs := env.References(env.RegexpSearch("lib/lib.go", "K"))
		checkLocations("References", refs, "lib/lib.go")

		impls := env.Implementations(env.RegexpSearch("lib/lib.go", "I"))
		checkLocations("Implementations", impls) // no implementations

		// Opening the standalone file should not result in any diagnostics.
		env.OpenFile("lib/ignore.go")
		env.AfterChange(NoDiagnostics())

		// Having opened the standalone file, we should find its symbols in the
		// workspace.
		syms = env.Symbol("K")
		if got, want := len(syms), 2; got != want {
			t.Fatalf("got %d symbols, want %d", got, want)
		}

		foundMainK := false
		var symNames []string
		for _, sym := range syms {
			symNames = append(symNames, sym.Name)
			if sym.Name == "main.K" {
				foundMainK = true
			}
		}
		if !foundMainK {
			t.Errorf("WorkspaceSymbol(\"K\") = %v, want containing main.K", symNames)
		}

		// We should resolve workspace definitions in the standalone file.
		fileLoc := env.GoToDefinition(env.RegexpSearch("lib/ignore.go", "lib.(K)"))
		file := env.Sandbox.Workdir.URIToPath(fileLoc.URI)
		if got, want := file, "lib/lib.go"; got != want {
			t.Errorf("GoToDefinition(lib.K) = %v, want %v", got, want)
		}

		// ...as well as intra-file definitions
		loc := env.GoToDefinition(env.RegexpSearch("lib/ignore.go", "\\+ (K)"))
		wantLoc := env.RegexpSearch("lib/ignore.go", "const (K)")
		if loc != wantLoc {
			t.Errorf("GoToDefinition(K) = %v, want %v", loc, wantLoc)
		}

		// Renaming "lib.K" to "lib.D" should cause a diagnostic in the standalone
		// file.
		env.RegexpReplace("lib/lib.go", "K", "D")
		env.AfterChange(Diagnostics(env.AtRegexp("lib/ignore.go", "lib.(K)")))

		// Undoing the replacement should fix diagnostics
		env.RegexpReplace("lib/lib.go", "D", "K")
		env.AfterChange(NoDiagnostics())

		// Now that our workspace has no errors, we should be able to find
		// references and rename.
		refs = env.References(env.RegexpSearch("lib/lib.go", "K"))
		checkLocations("References", refs, "lib/lib.go", "lib/ignore.go")

		impls = env.Implementations(env.RegexpSearch("lib/lib.go", "I"))
		checkLocations("Implementations", impls, "lib/ignore.go")

		// Renaming should rename in the standalone package.
		env.Rename(env.RegexpSearch("lib/lib.go", "K"), "D")
		env.RegexpSearch("lib/ignore.go", "lib.D")
	})
}

func TestStandaloneFiles_Configuration(t *testing.T) {
	const files = `
-- go.mod --
module mod.test

go 1.18
-- lib.go --
package lib // without this package, files are loaded as command-line-arguments
-- ignore.go --
//go:build ignore
// +build ignore

package main

// An arbitrary comment.

func main() {}
-- standalone.go --
//go:build standalone
// +build standalone

package main

func main() {}
`

	WithOptions(
		Settings{
			"standaloneTags": []string{"standalone", "script"},
		},
	).Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("ignore.go")
		env.OpenFile("standalone.go")

		env.AfterChange(
			Diagnostics(env.AtRegexp("ignore.go", "package (main)")),
			NoDiagnostics(ForFile("standalone.go")),
		)

		cfg := env.Editor.Config()
		cfg.Settings = map[string]interface{}{
			"standaloneTags": []string{"ignore"},
		}
		env.ChangeConfiguration(cfg)
		env.AfterChange(
			NoDiagnostics(ForFile("ignore.go")),
			Diagnostics(env.AtRegexp("standalone.go", "package (main)")),
		)
	})
}
