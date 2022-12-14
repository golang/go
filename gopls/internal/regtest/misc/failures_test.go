// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"testing"

	. "golang.org/x/tools/gopls/internal/lsp/regtest"
	"golang.org/x/tools/gopls/internal/lsp/tests/compare"
)

// This is a slight variant of TestHoverOnError in definition_test.go
// that includes a line directive, which makes no difference since
// gopls ignores line directives.
func TestHoverFailure(t *testing.T) {
	t.Skip("line directives //line ")
	const mod = `
-- go.mod --
module mod.com

go 1.12
-- a.y --
DWIM(main)

-- main.go --
//line a.y:1
package main

func main() {
	var err error
	err.Error()
}`
	Run(t, mod, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		content, _ := env.Hover("main.go", env.RegexpSearch("main.go", "Error"))
		if content == nil {
			t.Fatalf("Hover('Error') returned nil")
		}
		want := "```go\nfunc (error).Error() string\n```"
		if content.Value != want {
			t.Fatalf("wrong Hover('Error') content:\n%s", compare.Text(want, content.Value))
		}
	})
}

// This test demonstrates a case where gopls is not at all confused by
// line directives, because it completely ignores them.
func TestFailingDiagnosticClearingOnEdit(t *testing.T) {
	t.Skip("line directives //line ")
	// badPackageDup contains a duplicate definition of the 'a' const.
	// This is a minor variant of TestDiagnosticClearingOnEditfrom from
	// diagnostics_test.go, with a line directive, which makes no difference.
	const badPackageDup = `
-- go.mod --
module mod.com

go 1.12
-- a.go --
package consts

const a = 1
-- b.go --
package consts
//line gen.go:5
const a = 2
`

	Run(t, badPackageDup, func(t *testing.T, env *Env) {
		env.OpenFile("b.go")
		env.Await(
			env.DiagnosticAtRegexpWithMessage("b.go", `a = 2`, "a redeclared"),
			env.DiagnosticAtRegexpWithMessage("a.go", `a = 1`, "other declaration"),
		)

		// Fix the error by editing the const name in b.go to `b`.
		env.RegexpReplace("b.go", "(a) = 2", "b")
		env.Await(
			EmptyOrNoDiagnostics("a.go"),
			EmptyOrNoDiagnostics("b.go"),
		)
	})
}
