// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package diagnostics

import (
	"testing"

	"golang.org/x/tools/internal/lsp/protocol"
	. "golang.org/x/tools/internal/lsp/regtest"
)

func TestUndeclaredDiagnostics(t *testing.T) {
	src := `
-- go.mod --
module mod.com

go 1.12
-- a/a.go --
package a

func _() int {
	return x
}
-- b/b.go --
package b

func _() int {
	var y int
	y = y
	return y
}
`
	Run(t, src, func(t *testing.T, env *Env) {
		isUnnecessary := func(diag protocol.Diagnostic) bool {
			for _, tag := range diag.Tags {
				if tag == protocol.Unnecessary {
					return true
				}
			}
			return false
		}

		// 'x' is undeclared, but still necessary.
		env.OpenFile("a/a.go")
		env.Await(env.DiagnosticAtRegexp("a/a.go", "x"))
		diags := env.DiagnosticsFor("a/a.go")
		if got := len(diags.Diagnostics); got != 1 {
			t.Errorf("len(Diagnostics) = %d, want 1", got)
		}
		if diag := diags.Diagnostics[0]; isUnnecessary(diag) {
			t.Errorf("%v tagged unnecessary, want necessary", diag)
		}

		// 'y = y' is pointless, and should be detected as unnecessary.
		env.OpenFile("b/b.go")
		env.Await(env.DiagnosticAtRegexp("b/b.go", "y = y"))
		diags = env.DiagnosticsFor("b/b.go")
		if got := len(diags.Diagnostics); got != 1 {
			t.Errorf("len(Diagnostics) = %d, want 1", got)
		}
		if diag := diags.Diagnostics[0]; !isUnnecessary(diag) {
			t.Errorf("%v tagged necessary, want unnecessary", diag)
		}
	})
}
