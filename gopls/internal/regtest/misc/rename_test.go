// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"strings"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	. "golang.org/x/tools/gopls/internal/lsp/regtest"
	"golang.org/x/tools/internal/testenv"
)

func TestPrepareRenamePackage(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.18
-- main.go --
package main

import (
	"fmt"
)

func main() {
	fmt.Println(1)
}
`
	const wantErr = "can't rename packages: LSP client does not support file renaming"
	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		pos := env.RegexpSearch("main.go", `main`)
		tdpp := protocol.TextDocumentPositionParams{
			TextDocument: env.Editor.TextDocumentIdentifier("main.go"),
			Position:     pos.ToProtocolPosition(),
		}
		params := &protocol.PrepareRenameParams{
			TextDocumentPositionParams: tdpp,
		}
		_, err := env.Editor.Server.PrepareRename(env.Ctx, params)
		if err == nil {
			t.Errorf("missing can't rename package error from PrepareRename")
		}

		if err.Error() != wantErr {
			t.Errorf("got %v, want %v", err.Error(), wantErr)
		}
	})
}

func TestRenamePackageInRenamedPackage(t *testing.T) {
	// Failed at Go 1.13; not investigated
	testenv.NeedsGo1Point(t, 14)
	const files = `
-- go.mod --
module mod.com

go 1.18
-- main.go --
package main

import (
	"fmt"
	"a.go"
)

func main() {
	fmt.Println(a.C)
}
-- a.go --
package main

const C = 1
`
	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		pos := env.RegexpSearch("main.go", "main")
		env.Rename("main.go", pos, "pkg")

		// Check if the new package name exists.
		env.RegexpSearch("main.go", "package pkg")
		env.RegexpSearch("a.go", "package pkg")
	})
}

// Test for golang/go#47564.
func TestRenameInTestVariant(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.12
-- stringutil/stringutil.go --
package stringutil

func Identity(s string) string {
	return s
}
-- stringutil/stringutil_test.go --
package stringutil

func TestIdentity(t *testing.T) {
	if got := Identity("foo"); got != "foo" {
		t.Errorf("bad")
	}
}
-- main.go --
package main

import (
	"fmt"

	"mod.com/stringutil"
)

func main() {
	fmt.Println(stringutil.Identity("hello world"))
}
`

	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		pos := env.RegexpSearch("main.go", `stringutil\.(Identity)`)
		env.Rename("main.go", pos, "Identityx")
		text := env.Editor.BufferText("stringutil/stringutil_test.go")
		if !strings.Contains(text, "Identityx") {
			t.Errorf("stringutil/stringutil_test.go: missing expected token `Identityx` after rename:\n%s", text)
		}
	})
}

// This is a test that rename operation initiated by the editor function as expected.
func TestRenameFileFromEditor(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.16
-- a/a.go --
package a

const X = 1
-- a/x.go --
package a

const X = 2
-- b/b.go --
package b
`

	Run(t, files, func(t *testing.T, env *Env) {
		// Rename files and verify that diagnostics are affected accordingly.

		// Initially, we should have diagnostics on both X's, for their duplicate declaration.
		env.Await(
			OnceMet(
				InitialWorkspaceLoad,
				env.DiagnosticAtRegexp("a/a.go", "X"),
				env.DiagnosticAtRegexp("a/x.go", "X"),
			),
		)

		// Moving x.go should make the diagnostic go away.
		env.RenameFile("a/x.go", "b/x.go")
		env.Await(
			OnceMet(
				env.DoneWithChangeWatchedFiles(),
				EmptyDiagnostics("a/a.go"),                  // no more duplicate declarations
				env.DiagnosticAtRegexp("b/b.go", "package"), // as package names mismatch
			),
		)

		// Renaming should also work on open buffers.
		env.OpenFile("b/x.go")

		// Moving x.go back to a/ should cause the diagnostics to reappear.
		env.RenameFile("b/x.go", "a/x.go")
		// TODO(rfindley): enable using a OnceMet precondition here. We can't
		// currently do this because DidClose, DidOpen and DidChangeWatchedFiles
		// are sent, and it is not easy to use all as a precondition.
		env.Await(
			env.DiagnosticAtRegexp("a/a.go", "X"),
			env.DiagnosticAtRegexp("a/x.go", "X"),
		)

		// Renaming the entire directory should move both the open and closed file.
		env.RenameFile("a", "x")
		env.Await(
			env.DiagnosticAtRegexp("x/a.go", "X"),
			env.DiagnosticAtRegexp("x/x.go", "X"),
		)

		// As a sanity check, verify that x/x.go is open.
		if text := env.Editor.BufferText("x/x.go"); text == "" {
			t.Fatal("got empty buffer for x/x.go")
		}
	})
}
