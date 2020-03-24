// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"testing"

	"golang.org/x/tools/internal/lsp/fake"
)

// Use mod.com for all go.mod files due to golang/go#35230.
const exampleProgram = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

import "fmt"

func main() {
	fmt.Println("Hello World.")
}`

func TestDiagnosticErrorInEditedFile(t *testing.T) {
	runner.Run(t, exampleProgram, func(env *Env) {
		// Deleting the 'n' at the end of Println should generate a single error
		// diagnostic.
		env.OpenFile("main.go")
		env.RegexpReplace("main.go", "Printl(n)", "")
		env.Await(env.DiagnosticAtRegexp("main.go", "Printl"))
	})
}

const onlyMod = `
-- go.mod --
module mod.com

go 1.12
`

func TestMissingImportDiagsClearOnFirstFile(t *testing.T) {
	t.Skip("skipping due to golang.org/issues/37195")
	t.Parallel()
	runner.Run(t, onlyMod, func(env *Env) {
		env.CreateBuffer("main.go", "package main\n\nfunc m() {\nlog.Println()\n}")
		env.SaveBuffer("main.go")
		// TODO: this shouldn't actually happen
		env.Await(env.DiagnosticAtRegexp("main.go", "Println"))
	})
}

const brokenFile = `package main

const Foo = "abc
`

func TestDiagnosticErrorInNewFile(t *testing.T) {
	runner.Run(t, brokenFile, func(env *Env) {
		env.CreateBuffer("broken.go", brokenFile)
		env.Await(env.DiagnosticAtRegexp("broken.go", "\"abc"))
	})
}

// badPackage contains a duplicate definition of the 'a' const.
const badPackage = `
-- go.mod --
module mod.com

go 1.12
-- a.go --
package consts

const a = 1
-- b.go --
package consts

const a = 2
`

func TestDiagnosticClearingOnEdit(t *testing.T) {
	runner.Run(t, badPackage, func(env *Env) {
		env.OpenFile("b.go")
		env.Await(env.DiagnosticAtRegexp("a.go", "a = 1"), env.DiagnosticAtRegexp("b.go", "a = 2"))

		// Fix the error by editing the const name in b.go to `b`.
		env.RegexpReplace("b.go", "(a) = 2", "b")
		env.Await(EmptyDiagnostics("a.go"), EmptyDiagnostics("b.go"))
	})
}

func TestDiagnosticClearingOnDelete(t *testing.T) {
	runner.Run(t, badPackage, func(env *Env) {
		env.OpenFile("a.go")
		env.Await(env.DiagnosticAtRegexp("a.go", "a = 1"), env.DiagnosticAtRegexp("b.go", "a = 2"))
		env.RemoveFileFromWorkspace("b.go")

		env.Await(EmptyDiagnostics("a.go"), EmptyDiagnostics("b.go"))
	})
}

func TestDiagnosticClearingOnClose(t *testing.T) {
	runner.Run(t, badPackage, func(env *Env) {
		env.CreateBuffer("c.go", `package consts

const a = 3`)
		env.Await(
			env.DiagnosticAtRegexp("a.go", "a = 1"),
			env.DiagnosticAtRegexp("b.go", "a = 2"),
			env.DiagnosticAtRegexp("c.go", "a = 3"))
		env.CloseBuffer("c.go")
		env.Await(
			env.DiagnosticAtRegexp("a.go", "a = 1"),
			env.DiagnosticAtRegexp("b.go", "a = 2"),
			EmptyDiagnostics("c.go"))
	})
}

func TestIssue37978(t *testing.T) {
	runner.Run(t, exampleProgram, func(env *Env) {
		// Create a new workspace-level directory and empty file.
		env.CreateBuffer("c/c.go", "")

		// Write the file contents with a missing import.
		env.EditBuffer("c/c.go", fake.Edit{
			Text: `package c

const a = http.MethodGet
`,
		})
		env.Await(
			env.DiagnosticAtRegexp("c/c.go", "http.MethodGet"),
		)
		// Save file, which will organize imports, adding the expected import.
		// Expect the diagnostics to clear.
		env.SaveBuffer("c/c.go")
		env.Await(
			EmptyDiagnostics("c/c.go"),
		)
	})
}
