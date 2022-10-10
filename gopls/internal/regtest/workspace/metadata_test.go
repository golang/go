// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package workspace

import (
	"testing"

	. "golang.org/x/tools/gopls/internal/lsp/regtest"
	"golang.org/x/tools/internal/testenv"
)

// TODO(rfindley): move workspace tests related to metadata bugs into this
// file.

func TestFixImportDecl(t *testing.T) {
	// It appears that older Go versions don't even see p.go from the initial
	// workspace load.
	testenv.NeedsGo1Point(t, 15)
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
		env.Await(OnceMet(
			env.DoneWithChange(),
			EmptyDiagnostics("p.go"),
		))
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
		env.Await(
			OnceMet(
				env.DoneWithOpen(),
				env.DiagnosticAtRegexp("foo.go", "func (main)"),
				env.DiagnosticAtRegexp("bar.go", "func (main)"),
			),
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
		env.Await(
			OnceMet(
				env.DoneWithChange(),
				EmptyDiagnostics("foo.go"),
				EmptyDiagnostics("bar.go"),
			),
		)

		// If instead of 'ignore' (which gopls treats as a standalone package) we
		// used a different build tag, we should get a warning about having no
		// packages for bar.go
		env.RegexpReplace("bar.go", "ignore", "excluded")
		env.Await(
			OnceMet(
				env.DoneWithChange(),
				env.DiagnosticAtRegexpWithMessage("bar.go", "package (main)", "No packages"),
			),
		)
	})
}
