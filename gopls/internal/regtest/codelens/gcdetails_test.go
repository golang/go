// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codelens

import (
	"runtime"
	"strings"
	"testing"

	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/lsp/command"
	"golang.org/x/tools/gopls/internal/lsp/fake"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	. "golang.org/x/tools/gopls/internal/lsp/regtest"
)

func TestGCDetails_Toggle(t *testing.T) {
	if runtime.GOOS == "android" {
		t.Skipf("the gc details code lens doesn't work on Android")
	}

	const mod = `
-- go.mod --
module mod.com

go 1.15
-- main.go --
package main

import "fmt"

func main() {
	fmt.Println(42)
}
`
	WithOptions(
		Settings{
			"codelenses": map[string]bool{
				"gc_details": true,
			},
		},
	).Run(t, mod, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.ExecuteCodeLensCommand("main.go", command.GCDetails, nil)
		d := &protocol.PublishDiagnosticsParams{}
		env.OnceMet(
			Diagnostics(AtPosition("main.go", 5, 13)),
			ReadDiagnostics("main.go", d),
		)
		// Confirm that the diagnostics come from the gc details code lens.
		var found bool
		for _, d := range d.Diagnostics {
			if d.Severity != protocol.SeverityInformation {
				t.Fatalf("unexpected diagnostic severity %v, wanted Information", d.Severity)
			}
			if strings.Contains(d.Message, "42 escapes") {
				found = true
			}
		}
		if !found {
			t.Fatalf(`expected to find diagnostic with message "escape(42 escapes to heap)", found none`)
		}

		// Editing a buffer should cause gc_details diagnostics to disappear, since
		// they only apply to saved buffers.
		env.EditBuffer("main.go", fake.NewEdit(0, 0, 0, 0, "\n\n"))
		env.AfterChange(NoDiagnostics(ForFile("main.go")))

		// Saving a buffer should re-format back to the original state, and
		// re-enable the gc_details diagnostics.
		env.SaveBuffer("main.go")
		env.AfterChange(Diagnostics(AtPosition("main.go", 5, 13)))

		// Toggle the GC details code lens again so now it should be off.
		env.ExecuteCodeLensCommand("main.go", command.GCDetails, nil)
		env.Await(NoDiagnostics(ForFile("main.go")))
	})
}

// Test for the crasher in golang/go#54199
func TestGCDetails_NewFile(t *testing.T) {
	bug.PanicOnBugs = false
	const src = `
-- go.mod --
module mod.test

go 1.12
`

	WithOptions(
		Settings{
			"codelenses": map[string]bool{
				"gc_details": true,
			},
		},
	).Run(t, src, func(t *testing.T, env *Env) {
		env.CreateBuffer("p_test.go", "")

		const gcDetailsCommand = "gopls." + string(command.GCDetails)

		hasGCDetails := func() bool {
			lenses := env.CodeLens("p_test.go") // should not crash
			for _, lens := range lenses {
				if lens.Command.Command == gcDetailsCommand {
					return true
				}
			}
			return false
		}

		// With an empty file, we shouldn't get the gc_details codelens because
		// there is nowhere to position it (it needs a package name).
		if hasGCDetails() {
			t.Errorf("got the gc_details codelens for an empty file")
		}

		// Edit to provide a package name.
		env.EditBuffer("p_test.go", fake.NewEdit(0, 0, 0, 0, "package p"))

		// Now we should get the gc_details codelens.
		if !hasGCDetails() {
			t.Errorf("didn't get the gc_details codelens for a valid non-empty Go file")
		}
	})
}
