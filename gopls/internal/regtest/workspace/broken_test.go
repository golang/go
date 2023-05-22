// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package workspace

import (
	"strings"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp"
	. "golang.org/x/tools/gopls/internal/lsp/regtest"
	"golang.org/x/tools/internal/testenv"
)

// This file holds various tests for UX with respect to broken workspaces.
//
// TODO: consolidate other tests here.
//
// TODO: write more tests:
//  - an explicit GOWORK value that doesn't exist
//  - using modules and/or GOWORK inside of GOPATH?

// Test for golang/go#53933
func TestBrokenWorkspace_DuplicateModules(t *testing.T) {
	// The go command error message was improved in Go 1.20 to mention multiple
	// modules.
	testenv.NeedsGo1Point(t, 20)

	// This proxy module content is replaced by the workspace, but is still
	// required for module resolution to function in the Go command.
	const proxy = `
-- example.com/foo@v0.0.1/go.mod --
module example.com/foo

go 1.12
`

	const src = `
-- go.work --
go 1.18

use (
	./package1
	./package1/vendor/example.com/foo
	./package2
	./package2/vendor/example.com/foo
)

-- package1/go.mod --
module mod.test

go 1.18

require example.com/foo v0.0.1
-- package1/main.go --
package main

import "example.com/foo"

func main() {
	_ = foo.CompleteMe
}
-- package1/vendor/example.com/foo/go.mod --
module example.com/foo

go 1.18
-- package1/vendor/example.com/foo/foo.go --
package foo

const CompleteMe = 111
-- package2/go.mod --
module mod2.test

go 1.18

require example.com/foo v0.0.1
-- package2/main.go --
package main

import "example.com/foo"

func main() {
	_ = foo.CompleteMe
}
-- package2/vendor/example.com/foo/go.mod --
module example.com/foo

go 1.18
-- package2/vendor/example.com/foo/foo.go --
package foo

const CompleteMe = 222
`

	WithOptions(
		ProxyFiles(proxy),
	).Run(t, src, func(t *testing.T, env *Env) {
		env.OpenFile("package1/main.go")
		env.AfterChange(
			OutstandingWork(lsp.WorkspaceLoadFailure, `module example.com/foo appears multiple times in workspace`),
		)

		// Remove the redundant vendored copy of example.com.
		env.WriteWorkspaceFile("go.work", `go 1.18
		use (
			./package1
			./package2
			./package2/vendor/example.com/foo
		)
		`)
		env.AfterChange(NoOutstandingWork())

		// Check that definitions in package1 go to the copy vendored in package2.
		location := string(env.GoToDefinition(env.RegexpSearch("package1/main.go", "CompleteMe")).URI)
		const wantLocation = "package2/vendor/example.com/foo/foo.go"
		if !strings.HasSuffix(location, wantLocation) {
			t.Errorf("got definition of CompleteMe at %q, want %q", location, wantLocation)
		}
	})
}

// Test for golang/go#43186: correcting the module path should fix errors
// without restarting gopls.
func TestBrokenWorkspace_WrongModulePath(t *testing.T) {
	const files = `
-- go.mod --
module mod.testx

go 1.18
-- p/internal/foo/foo.go --
package foo

const C = 1
-- p/internal/bar/bar.go --
package bar

import "mod.test/p/internal/foo"

const D = foo.C + 1
-- p/internal/bar/bar_test.go --
package bar_test

import (
	"mod.test/p/internal/foo"
	. "mod.test/p/internal/bar"
)

const E = D + foo.C
-- p/internal/baz/baz_test.go --
package baz_test

import (
	named "mod.test/p/internal/bar"
)

const F = named.D - 3
`

	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("p/internal/bar/bar.go")
		env.AfterChange(
			Diagnostics(env.AtRegexp("p/internal/bar/bar.go", "\"mod.test/p/internal/foo\"")),
		)
		env.OpenFile("go.mod")
		env.RegexpReplace("go.mod", "mod.testx", "mod.test")
		env.SaveBuffer("go.mod") // saving triggers a reload
		env.AfterChange(NoDiagnostics())
	})
}

func TestMultipleModules_Warning(t *testing.T) {
	msgForVersion := func(ver int) string {
		if ver >= 18 {
			return `gopls was not able to find modules in your workspace.`
		} else {
			return `gopls requires a module at the root of your workspace.`
		}
	}

	const modules = `
-- a/go.mod --
module a.com

go 1.12
-- a/a.go --
package a
-- a/empty.go --
// an empty file
-- b/go.mod --
module b.com

go 1.12
-- b/b.go --
package b
`
	for _, go111module := range []string{"on", "auto"} {
		t.Run("GO111MODULE="+go111module, func(t *testing.T) {
			WithOptions(
				Modes(Default),
				EnvVars{"GO111MODULE": go111module},
			).Run(t, modules, func(t *testing.T, env *Env) {
				ver := env.GoVersion()
				msg := msgForVersion(ver)
				env.OpenFile("a/a.go")
				env.OpenFile("a/empty.go")
				env.OpenFile("b/go.mod")
				env.AfterChange(
					Diagnostics(env.AtRegexp("a/a.go", "package a")),
					Diagnostics(env.AtRegexp("b/go.mod", "module b.com")),
					OutstandingWork(lsp.WorkspaceLoadFailure, msg),
				)

				// Changing the workspace folders to the valid modules should resolve
				// the workspace errors and diagnostics.
				//
				// TODO(rfindley): verbose work tracking doesn't follow changing the
				// workspace folder, therefore we can't invoke AfterChange here.
				env.ChangeWorkspaceFolders("a", "b")
				env.Await(
					NoDiagnostics(ForFile("a/a.go")),
					NoDiagnostics(ForFile("b/go.mod")),
					NoOutstandingWork(),
				)

				env.ChangeWorkspaceFolders(".")

				// TODO(rfindley): when GO111MODULE=auto, we need to open or change a
				// file here in order to detect a critical error. This is because gopls
				// has forgotten about a/a.go, and therefore doesn't hit the heuristic
				// "all packages are command-line-arguments".
				//
				// This is broken, and could be fixed by adjusting the heuristic to
				// account for the scenario where there are *no* workspace packages, or
				// (better) trying to get workspace packages for each open file. See
				// also golang/go#54261.
				env.OpenFile("b/b.go")
				env.AfterChange(
					// TODO(rfindley): fix these missing diagnostics.
					// Diagnostics(env.AtRegexp("a/a.go", "package a")),
					// Diagnostics(env.AtRegexp("b/go.mod", "module b.com")),
					Diagnostics(env.AtRegexp("b/b.go", "package b")),
					OutstandingWork(lsp.WorkspaceLoadFailure, msg),
				)
			})
		})
	}

	// Expect no warning if GO111MODULE=auto in a directory in GOPATH.
	t.Run("GOPATH_GO111MODULE_auto", func(t *testing.T) {
		WithOptions(
			Modes(Default),
			EnvVars{"GO111MODULE": "auto"},
			InGOPATH(),
		).Run(t, modules, func(t *testing.T, env *Env) {
			env.OpenFile("a/a.go")
			env.AfterChange(
				NoDiagnostics(ForFile("a/a.go")),
				NoOutstandingWork(),
			)
		})
	})
}
