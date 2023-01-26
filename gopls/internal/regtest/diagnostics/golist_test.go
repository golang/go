// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package diagnostics

import (
	"testing"

	. "golang.org/x/tools/gopls/internal/lsp/regtest"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/internal/testenv"
)

func TestGoListErrors(t *testing.T) {
	testenv.NeedsTool(t, "cgo")

	const src = `
-- go.mod --
module a.com

go 1.18
-- a/a.go --
package a

import
-- c/c.go --
package c

/*
int fortythree() { return 42; }
*/
import "C"

func Foo() {
	print(C.fortytwo())
}
-- p/p.go --
package p

import "a.com/q"

const P = q.Q + 1
-- q/q.go --
package q

import "a.com/p"

const Q = p.P + 1
`

	Run(t, src, func(t *testing.T, env *Env) {
		env.OnceMet(
			InitialWorkspaceLoad,
			Diagnostics(
				env.AtRegexp("a/a.go", "import\n()"),
				FromSource(string(source.ParseError)),
			),
			Diagnostics(
				AtPosition("c/c.go", 0, 0),
				FromSource(string(source.ListError)),
				WithMessage("may indicate failure to perform cgo processing"),
			),
			Diagnostics(
				env.AtRegexp("p/p.go", `"a.com/q"`),
				FromSource(string(source.ListError)),
				WithMessage("import cycle not allowed"),
			),
		)
	})
}
