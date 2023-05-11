// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package workspace

import (
	"strings"
	"testing"

	. "golang.org/x/tools/gopls/internal/lsp/regtest"
	"golang.org/x/tools/internal/testenv"
)

// TODO(rfindley): move workspace tests related to metadata bugs into this
// file.

func TestFixImportDecl(t *testing.T) {
	const src = `
-- go.mod --
module mod.test

go 1.12
-- p.go --
package p

import (
	_ "fmt"

const C = 42
`

	Run(t, src, func(t *testing.T, env *Env) {
		env.OpenFile("p.go")
		env.RegexpReplace("p.go", "\"fmt\"", "\"fmt\"\n)")
		env.AfterChange(
			NoDiagnostics(ForFile("p.go")),
		)
	})
}

// Test that moving ignoring a file via build constraints causes diagnostics to
// be resolved.
func TestIgnoreFile(t *testing.T) {
	testenv.NeedsGo1Point(t, 17) // needs native overlays and support for go:build directives

	const src = `
-- go.mod --
module mod.test

go 1.12
-- foo.go --
package main

func main() {}
-- bar.go --
package main

func main() {}
	`

	WithOptions(
		// TODO(golang/go#54180): we don't run in 'experimental' mode here, because
		// with "experimentalUseInvalidMetadata", this test fails because the
		// orphaned bar.go is diagnosed using stale metadata, and then not
		// re-diagnosed when new metadata arrives.
		//
		// We could fix this by re-running diagnostics after a load, but should
		// consider whether that is worthwhile.
		Modes(Default),
	).Run(t, src, func(t *testing.T, env *Env) {
		env.OpenFile("foo.go")
		env.OpenFile("bar.go")
		env.OnceMet(
			env.DoneWithOpen(),
			Diagnostics(env.AtRegexp("foo.go", "func (main)")),
			Diagnostics(env.AtRegexp("bar.go", "func (main)")),
		)

		// Ignore bar.go. This should resolve diagnostics.
		env.RegexpReplace("bar.go", "package main", "//go:build ignore\n\npackage main")

		// To make this test pass with experimentalUseInvalidMetadata, we could make
		// an arbitrary edit that invalidates the snapshot, at which point the
		// orphaned diagnostics will be invalidated.
		//
		// But of course, this should not be necessary: we should invalidate stale
		// information when fresh metadata arrives.
		// env.RegexpReplace("foo.go", "package main", "package main // test")
		env.AfterChange(
			NoDiagnostics(ForFile("foo.go")),
			NoDiagnostics(ForFile("bar.go")),
		)

		// If instead of 'ignore' (which gopls treats as a standalone package) we
		// used a different build tag, we should get a warning about having no
		// packages for bar.go
		env.RegexpReplace("bar.go", "ignore", "excluded")
		env.AfterChange(
			Diagnostics(env.AtRegexp("bar.go", "package (main)"), WithMessage("not included in your workspace")),
		)
	})
}

func TestReinitializeRepeatedly(t *testing.T) {
	testenv.NeedsGo1Point(t, 18) // uses go.work

	const multiModule = `
-- go.work --
go 1.18

use (
	moda/a
	modb
)
-- moda/a/go.mod --
module a.com

require b.com v1.2.3
-- moda/a/go.sum --
b.com v1.2.3 h1:tXrlXP0rnjRpKNmkbLYoWBdq0ikb3C3bKK9//moAWBI=
b.com v1.2.3/go.mod h1:D+J7pfFBZK5vdIdZEFquR586vKKIkqG7Qjw9AxG5BQ8=
-- moda/a/a.go --
package a

import (
	"b.com/b"
)

func main() {
	var x int
	_ = b.Hello()
	// AAA
}
-- modb/go.mod --
module b.com

-- modb/b/b.go --
package b

func Hello() int {
	var x int
}
`
	WithOptions(
		ProxyFiles(workspaceModuleProxy),
		Settings{
			// For this test, we want workspace diagnostics to start immediately
			// during change processing.
			"diagnosticsDelay": "0",
		},
	).Run(t, multiModule, func(t *testing.T, env *Env) {
		env.OpenFile("moda/a/a.go")
		env.AfterChange()

		// This test verifies that we fully process workspace reinitialization
		// (which allows GOPROXY), even when the reinitialized snapshot is
		// invalidated by subsequent changes.
		//
		// First, update go.work to remove modb. This will cause reinitialization
		// to fetch b.com from the proxy.
		env.WriteWorkspaceFile("go.work", "go 1.18\nuse moda/a")
		// Next, wait for gopls to start processing the change. Because we've set
		// diagnosticsDelay to zero, this will start diagnosing the workspace (and
		// try to reinitialize on the snapshot context).
		env.Await(env.StartedChangeWatchedFiles())
		// Finally, immediately make a file change to cancel the previous
		// operation. This is racy, but will usually cause initialization to be
		// canceled.
		env.RegexpReplace("moda/a/a.go", "AAA", "BBB")
		env.AfterChange()
		// Now, to satisfy a definition request, gopls will try to reload moda. But
		// without access to the proxy (because this is no longer a
		// reinitialization), this loading will fail.
		loc := env.GoToDefinition(env.RegexpSearch("moda/a/a.go", "Hello"))
		got := env.Sandbox.Workdir.URIToPath(loc.URI)
		if want := "b.com@v1.2.3/b/b.go"; !strings.HasSuffix(got, want) {
			t.Errorf("expected %s, got %v", want, got)
		}
	})
}

// Test for golang/go#59458. With lazy module loading, we may not need
// transitively required modules.
func TestNestedModuleLoading_Issue59458(t *testing.T) {
	testenv.NeedsGo1Point(t, 17) // needs lazy module loading

	// In this test, module b.com/nested requires b.com/other, which in turn
	// requires b.com, but b.com/nested does not reach b.com through the package
	// graph. Therefore, b.com/nested does not need b.com on 1.17 and later,
	// thanks to graph pruning.
	//
	// We verify that we can load b.com/nested successfully. Previously, we
	// couldn't, because loading the pattern b.com/nested/... matched the module
	// b.com, which exists in the module graph but does not have a go.sum entry.

	const proxy = `
-- b.com@v1.2.3/go.mod --
module b.com

go 1.18
-- b.com@v1.2.3/b/b.go --
package b

func Hello() {}

-- b.com/other@v1.4.6/go.mod --
module b.com/other

go 1.18

require b.com v1.2.3
-- b.com/other@v1.4.6/go.sun --
b.com v1.2.3 h1:AGjCxWRJLUuJiZ21IUTByr9buoa6+B6Qh5LFhVLKpn4=
-- b.com/other@v1.4.6/bar/bar.go --
package bar

import "b.com/b"

func _() {
	b.Hello()
}
-- b.com/other@v1.4.6/foo/foo.go --
package foo

const Foo = 0
`

	const files = `
-- go.mod --
module b.com/nested

go 1.18

require b.com/other v1.4.6
-- go.sum --
b.com/other v1.4.6 h1:pHXSzGsk6DamYXp9uRdDB9A/ZQqAN9it+JudU0sBf94=
b.com/other v1.4.6/go.mod h1:T0TYuGdAHw4p/l0+1P/yhhYHfZRia7PaadNVDu58OWM=
-- nested.go --
package nested

import "b.com/other/foo"

const C = foo.Foo
`
	WithOptions(
		ProxyFiles(proxy),
	).Run(t, files, func(t *testing.T, env *Env) {
		env.OnceMet(
			InitialWorkspaceLoad,
			NoDiagnostics(),
		)
	})
}
