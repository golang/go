// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"testing"

	"golang.org/x/tools/internal/lsp/tests"
	"golang.org/x/tools/internal/testenv"
)

func TestModFileModification(t *testing.T) {
	testenv.NeedsGo1Point(t, 14)

	const proxy = `
-- example.com@v1.2.3/go.mod --
module example.com

go 1.12
-- example.com@v1.2.3/blah/blah.go --
package blah

const Name = "Blah"
`
	const untidyModule = `
-- go.mod --
module mod.com

-- main.go --
package main

import "example.com/blah"

func main() {
	fmt.Println(blah.Name)
}`
	runner.Run(t, untidyModule, func(t *testing.T, env *Env) {
		// Open the file and make sure that the initial workspace load does not
		// modify the go.mod file.
		goModContent := env.ReadWorkspaceFile("go.mod")
		env.OpenFile("main.go")
		env.Await(
			env.DiagnosticAtRegexp("main.go", "\"example.com/blah\""),
		)
		if got := env.ReadWorkspaceFile("go.mod"); got != goModContent {
			t.Fatalf("go.mod changed on disk:\n%s", tests.Diff(want, got))
		}
		// Save the buffer, which will format and organize imports.
		// Confirm that the go.mod file still does not change.
		env.SaveBuffer("main.go")
		env.Await(
			env.DiagnosticAtRegexp("main.go", "\"example.com/blah\""),
		)
		if got := env.ReadWorkspaceFile("go.mod"); got != goModContent {
			t.Fatalf("go.mod changed on disk:\n%s", tests.Diff(want, got))
		}
	}, WithProxy(proxy))
}
