// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package misc

import (
	"testing"

	. "golang.org/x/tools/gopls/internal/lsp/regtest"
)

func TestMissingPatternDiagnostic(t *testing.T) {
	const files = `
-- go.mod --
module example.com
-- x.go --
package x

import (
	_ "embed"
)

// Issue 47436
func F() {}

//go:embed NONEXISTENT
var foo string
`
	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("x.go")
		env.Await(env.DiagnosticAtRegexpWithMessage("x.go", `NONEXISTENT`, "no matching files found"))
		env.RegexpReplace("x.go", `NONEXISTENT`, "x.go")
		env.Await(EmptyDiagnostics("x.go"))
	})
}
