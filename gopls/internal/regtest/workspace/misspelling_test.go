// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package workspace

import (
	"runtime"
	"testing"

	. "golang.org/x/tools/gopls/internal/lsp/regtest"
	"golang.org/x/tools/gopls/internal/lsp/tests/compare"
)

// Test for golang/go#57081.
func TestFormattingMisspelledURI(t *testing.T) {
	if runtime.GOOS != "windows" && runtime.GOOS != "darwin" {
		t.Skip("golang/go#57081 only reproduces on case-insensitive filesystems.")
	}
	const files = `
-- go.mod --
module mod.test

go 1.19
-- foo.go --
package foo

const  C = 2 // extra space is intentional
`

	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("Foo.go")
		env.FormatBuffer("Foo.go")
		want := env.BufferText("Foo.go")

		if want == "" {
			t.Fatalf("Foo.go is empty")
		}

		// In golang/go#57081, we observed that if overlay cases don't match, gopls
		// will find (and format) the on-disk contents rather than the overlay,
		// resulting in invalid edits.
		//
		// Verify that this doesn't happen, by confirming that formatting is
		// idempotent.
		env.FormatBuffer("Foo.go")
		got := env.BufferText("Foo.go")
		if diff := compare.Text(want, got); diff != "" {
			t.Errorf("invalid content after second formatting:\n%s", diff)
		}
	})
}

// Test that we can find packages for open files with different spelling on
// case-insensitive file systems.
func TestPackageForMisspelledURI(t *testing.T) {
	t.Skip("golang/go#57081: this test fails because the Go command does not load Foo.go correctly")
	if runtime.GOOS != "windows" && runtime.GOOS != "darwin" {
		t.Skip("golang/go#57081 only reproduces on case-insensitive filesystems.")
	}
	const files = `
-- go.mod --
module mod.test

go 1.19
-- foo.go --
package foo

const C = D
-- bar.go --
package foo

const D = 2
`

	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("Foo.go")
		env.AfterChange(NoDiagnostics())
	})
}
