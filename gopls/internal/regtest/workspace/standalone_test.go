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

const C = 0

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

const C = 1

type Mer struct{}
func (Mer) M()

func main() {
	println(lib.C + C)
}
`
	WithOptions(
		// On Go 1.17 and earlier, this test fails with
		// experimentalWorkspaceModule. Not investigated, as
		// experimentalWorkspaceModule will be removed.
		Modes(Default),
	).Run(t, files, func(t *testing.T, env *Env) {
		// Initially, gopls should not know about the standalone file as it hasn't
		// been opened. Therefore, we should only find one symbol 'C'.
		syms := env.Symbol("C")
		if got, want := len(syms), 1; got != want {
			t.Errorf("got %d symbols, want %d", got, want)
		}

		// Similarly, we should only find one reference to "C", and no
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

		// Replacing C with D should not cause any workspace diagnostics, since we
		// haven't yet opened the standalone file.
		env.RegexpReplace("lib/lib.go", "C", "D")
		env.AfterChange(NoDiagnostics())
		env.RegexpReplace("lib/lib.go", "D", "C")
		env.AfterChange(NoDiagnostics())

		refs := env.References(env.RegexpSearch("lib/lib.go", "C"))
		checkLocations("References", refs, "lib/lib.go")

		impls := env.Implementations(env.RegexpSearch("lib/lib.go", "I"))
		checkLocations("Implementations", impls) // no implementations

		// Opening the standalone file should not result in any diagnostics.
		env.OpenFile("lib/ignore.go")
		env.AfterChange(NoDiagnostics())

		// Having opened the standalone file, we should find its symbols in the
		// workspace.
		syms = env.Symbol("C")
		if got, want := len(syms), 2; got != want {
			t.Fatalf("got %d symbols, want %d", got, want)
		}

		foundMainC := false
		var symNames []string
		for _, sym := range syms {
			symNames = append(symNames, sym.Name)
			if sym.Name == "main.C" {
				foundMainC = true
			}
		}
		if !foundMainC {
			t.Errorf("WorkspaceSymbol(\"C\") = %v, want containing main.C", symNames)
		}

		// We should resolve workspace definitions in the standalone file.
		fileLoc := env.GoToDefinition(env.RegexpSearch("lib/ignore.go", "lib.(C)"))
		file := env.Sandbox.Workdir.URIToPath(fileLoc.URI)
		if got, want := file, "lib/lib.go"; got != want {
			t.Errorf("GoToDefinition(lib.C) = %v, want %v", got, want)
		}

		// ...as well as intra-file definitions
		loc := env.GoToDefinition(env.RegexpSearch("lib/ignore.go", "\\+ (C)"))
		wantLoc := env.RegexpSearch("lib/ignore.go", "const (C)")
		if loc != wantLoc {
			t.Errorf("GoToDefinition(C) = %v, want %v", loc, wantLoc)
		}

		// Renaming "lib.C" to "lib.D" should cause a diagnostic in the standalone
		// file.
		env.RegexpReplace("lib/lib.go", "C", "D")
		env.AfterChange(Diagnostics(env.AtRegexp("lib/ignore.go", "lib.(C)")))

		// Undoing the replacement should fix diagnostics
		env.RegexpReplace("lib/lib.go", "D", "C")
		env.AfterChange(NoDiagnostics())

		// Now that our workspace has no errors, we should be able to find
		// references and rename.
		refs = env.References(env.RegexpSearch("lib/lib.go", "C"))
		checkLocations("References", refs, "lib/lib.go", "lib/ignore.go")

		impls = env.Implementations(env.RegexpSearch("lib/lib.go", "I"))
		checkLocations("Implementations", impls, "lib/ignore.go")

		// Renaming should rename in the standalone package.
		env.Rename(env.RegexpSearch("lib/lib.go", "C"), "D")
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

		// TODO(golang/go#56158): gopls does not purge previously published
		// diagnostice when configuration changes.
		env.RegexpReplace("ignore.go", "arbitrary", "meaningless")

		env.AfterChange(
			NoDiagnostics(ForFile("ignore.go")),
			Diagnostics(env.AtRegexp("standalone.go", "package (main)")),
		)
	})
}
