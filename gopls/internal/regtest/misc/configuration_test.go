// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"testing"

	. "golang.org/x/tools/gopls/internal/regtest"

	"golang.org/x/tools/internal/lsp/fake"
	"golang.org/x/tools/internal/testenv"
)

// Test that enabling and disabling produces the expected results of showing
// and hiding staticcheck analysis results.
func TestChangeConfiguration(t *testing.T) {
	// Staticcheck only supports Go versions > 1.14.
	testenv.NeedsGo1Point(t, 15)

	const files = `
-- go.mod --
module mod.com

go 1.12
-- a/a.go --
package a

// NotThisVariable should really start with ThisVariable.
const ThisVariable = 7
`
	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("a/a.go")
		env.Await(
			env.DoneWithOpen(),
			NoDiagnostics("a/a.go"),
		)
		cfg := &fake.EditorConfig{}
		*cfg = env.Editor.Config
		cfg.EnableStaticcheck = true
		env.ChangeConfiguration(t, cfg)
		env.Await(
			DiagnosticAt("a/a.go", 2, 0),
		)
	})
}
