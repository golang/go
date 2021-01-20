// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"testing"

	"golang.org/x/tools/internal/lsp/fake"
)

// Test that enabling and disabling produces the expected results of showing
// and hiding staticcheck analysis results.
func TestChangeConfiguration(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.12
-- a/a.go --
package a

// NotThisVariable should really start with ThisVariable.
const ThisVariable = 7
`
	run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("a/a.go")
		env.Await(
			env.DoneWithOpen(),
			NoDiagnostics("a/a.go"),
		)
		cfg := &fake.EditorConfig{}
		*cfg = env.Editor.Config
		cfg.EnableStaticcheck = true
		env.changeConfiguration(t, cfg)
		env.Await(
			DiagnosticAt("a/a.go", 2, 0),
		)
	})
}
