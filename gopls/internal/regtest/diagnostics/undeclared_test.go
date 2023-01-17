// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package diagnostics

import (
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	. "golang.org/x/tools/gopls/internal/lsp/regtest"
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
		var adiags protocol.PublishDiagnosticsParams
		env.AfterChange(
			Diagnostics(env.AtRegexp("a/a.go", "x")),
			ReadDiagnostics("a/a.go", &adiags),
		)
		if got := len(adiags.Diagnostics); got != 1 {
			t.Errorf("len(Diagnostics) = %d, want 1", got)
		}
		if diag := adiags.Diagnostics[0]; isUnnecessary(diag) {
			t.Errorf("%v tagged unnecessary, want necessary", diag)
		}

		// 'y = y' is pointless, and should be detected as unnecessary.
		env.OpenFile("b/b.go")
		var bdiags protocol.PublishDiagnosticsParams
		env.AfterChange(
			Diagnostics(env.AtRegexp("b/b.go", "y = y")),
			ReadDiagnostics("b/b.go", &bdiags),
		)
		if got := len(bdiags.Diagnostics); got != 1 {
			t.Errorf("len(Diagnostics) = %d, want 1", got)
		}
		if diag := bdiags.Diagnostics[0]; !isUnnecessary(diag) {
			t.Errorf("%v tagged necessary, want unnecessary", diag)
		}
	})
}
