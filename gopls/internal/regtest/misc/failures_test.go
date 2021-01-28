// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"log"
	"testing"

	. "golang.org/x/tools/gopls/internal/regtest"
)

// This test passes (TestHoverOnError in definition_test.go) without
// the //line directive
func TestHoverFailure(t *testing.T) {
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
	WithOptions(SkipLogs()).Run(t, mod, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		content, _ := env.Hover("main.go", env.RegexpSearch("main.go", "Error"))
		// without the //line comment content would be non-nil
		if content != nil {
			t.Fatalf("expected nil hover content for Error")
		}
	})
}

// badPackageDup contains a duplicate definition of the 'a' const.
// this is from diagnostics_test.go,
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

func TestFailingDiagnosticClearingOnEdit(t *testing.T) {
	Run(t, badPackageDup, func(t *testing.T, env *Env) {
		log.SetFlags(log.Lshortfile)
		env.OpenFile("b.go")
		env.Await(env.AnyDiagnosticAtCurrentVersion("a.go"))
		// no diagnostics for either b.go or 'gen.go', but there should be
		env.Await(NoDiagnostics("b.go"))

		// Fix the error by editing the const name in b.go to `b`.
		env.RegexpReplace("b.go", "(a) = 2", "b")
		env.Await(
			EmptyDiagnostics("a.go"),
			// expected, as there have never been any diagnostics for b.go
			NoDiagnostics("b.go"),
		)
	})
}
