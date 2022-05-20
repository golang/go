// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package workspace

import (
	"testing"

	. "golang.org/x/tools/internal/lsp/regtest"
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
