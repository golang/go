// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package workspace

import (
	"sort"
	"strings"
	"testing"

	. "golang.org/x/tools/gopls/internal/lsp/regtest"
)

// This file contains regression tests for the directoryFilters setting.
//
// TODO:
//  - consolidate some of these tests into a single test
//  - add more tests for changing directory filters

func TestDirectoryFilters(t *testing.T) {
	WithOptions(
		ProxyFiles(workspaceProxy),
		WorkspaceFolders("pkg"),
		Settings{
			"directoryFilters": []string{"-inner"},
		},
	).Run(t, workspaceModule, func(t *testing.T, env *Env) {
		syms := env.WorkspaceSymbol("Hi")
		sort.Slice(syms, func(i, j int) bool { return syms[i].ContainerName < syms[j].ContainerName })
		for _, s := range syms {
			if strings.Contains(s.ContainerName, "inner") {
				t.Errorf("WorkspaceSymbol: found symbol %q with container %q, want \"inner\" excluded", s.Name, s.ContainerName)
			}
		}
	})
}

func TestDirectoryFiltersLoads(t *testing.T) {
	// exclude, and its error, should be excluded from the workspace.
	const files = `
-- go.mod --
module example.com

go 1.12
-- exclude/exclude.go --
package exclude

const _ = Nonexistant
`

	WithOptions(
		Settings{"directoryFilters": []string{"-exclude"}},
	).Run(t, files, func(t *testing.T, env *Env) {
		env.Await(NoDiagnostics("exclude/x.go"))
	})
}

func TestDirectoryFiltersTransitiveDep(t *testing.T) {
	// Even though exclude is excluded from the workspace, it should
	// still be importable as a non-workspace package.
	const files = `
-- go.mod --
module example.com

go 1.12
-- include/include.go --
package include
import "example.com/exclude"

const _ = exclude.X
-- exclude/exclude.go --
package exclude

const _ = Nonexistant // should be ignored, since this is a non-workspace package
const X = 1
`

	WithOptions(
		Settings{"directoryFilters": []string{"-exclude"}},
	).Run(t, files, func(t *testing.T, env *Env) {
		env.Await(
			NoDiagnostics("exclude/exclude.go"), // filtered out
			NoDiagnostics("include/include.go"), // successfully builds
		)
	})
}

func TestDirectoryFiltersWorkspaceModules(t *testing.T) {
	// Define a module include.com which should be in the workspace, plus a
	// module exclude.com which should be excluded and therefore come from
	// the proxy.
	const files = `
-- include/go.mod --
module include.com

go 1.12

require exclude.com v1.0.0

-- include/go.sum --
exclude.com v1.0.0 h1:Q5QSfDXY5qyNCBeUiWovUGqcLCRZKoTs9XdBeVz+w1I=
exclude.com v1.0.0/go.mod h1:hFox2uDlNB2s2Jfd9tHlQVfgqUiLVTmh6ZKat4cvnj4=

-- include/include.go --
package include

import "exclude.com"

var _ = exclude.X // satisfied only by the workspace version
-- exclude/go.mod --
module exclude.com

go 1.12
-- exclude/exclude.go --
package exclude

const X = 1
`
	const proxy = `
-- exclude.com@v1.0.0/go.mod --
module exclude.com

go 1.12
-- exclude.com@v1.0.0/exclude.go --
package exclude
`
	WithOptions(
		Modes(Experimental),
		ProxyFiles(proxy),
		Settings{"directoryFilters": []string{"-exclude"}},
	).Run(t, files, func(t *testing.T, env *Env) {
		env.Await(env.DiagnosticAtRegexp("include/include.go", `exclude.(X)`))
	})
}

// Test for golang/go#46438: support for '**' in directory filters.
func TestDirectoryFilters_Wildcard(t *testing.T) {
	filters := []string{"-**/bye"}
	WithOptions(
		ProxyFiles(workspaceProxy),
		WorkspaceFolders("pkg"),
		Settings{
			"directoryFilters": filters,
		},
	).Run(t, workspaceModule, func(t *testing.T, env *Env) {
		syms := env.WorkspaceSymbol("Bye")
		sort.Slice(syms, func(i, j int) bool { return syms[i].ContainerName < syms[j].ContainerName })
		for _, s := range syms {
			if strings.Contains(s.ContainerName, "bye") {
				t.Errorf("WorkspaceSymbol: found symbol %q with container %q with filters %v", s.Name, s.ContainerName, filters)
			}
		}
	})
}

// Test for golang/go#52993: wildcard directoryFilters should apply to
// goimports scanning as well.
func TestDirectoryFilters_ImportScanning(t *testing.T) {
	const files = `
-- go.mod --
module mod.test

go 1.12
-- main.go --
package main

func main() {
	bye.Goodbye()
}
-- p/bye/bye.go --
package bye

func Goodbye() {}
`

	WithOptions(
		Settings{
			"directoryFilters": []string{"-**/bye"},
		},
		// This test breaks in 'Experimental' mode, because with
		// experimentalWorkspaceModule set we the goimports scan behaves
		// differently.
		//
		// Since this feature is going away (golang/go#52897), don't investigate.
		Modes(Default),
	).Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		beforeSave := env.Editor.BufferText("main.go")
		env.OrganizeImports("main.go")
		got := env.Editor.BufferText("main.go")
		if got != beforeSave {
			t.Errorf("after organizeImports code action, got modified buffer:\n%s", got)
		}
	})
}

// Test for golang/go#52993: non-wildcard directoryFilters should still be
// applied relative to the workspace folder, not the module root.
func TestDirectoryFilters_MultiRootImportScanning(t *testing.T) {
	const files = `
-- go.work --
go 1.18

use (
	a
	b
)
-- a/go.mod --
module mod1.test

go 1.18
-- a/main.go --
package main

func main() {
	hi.Hi()
}
-- a/hi/hi.go --
package hi

func Hi() {}
-- b/go.mod --
module mod2.test

go 1.18
-- b/main.go --
package main

func main() {
	hi.Hi()
}
-- b/hi/hi.go --
package hi

func Hi() {}
`

	WithOptions(
		Settings{
			"directoryFilters": []string{"-hi"}, // this test fails with -**/hi
		},
	).Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("a/main.go")
		beforeSave := env.Editor.BufferText("a/main.go")
		env.OrganizeImports("a/main.go")
		got := env.Editor.BufferText("a/main.go")
		if got == beforeSave {
			t.Errorf("after organizeImports code action, got identical buffer:\n%s", got)
		}
	})
}
