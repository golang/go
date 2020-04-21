// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"testing"

	"golang.org/x/tools/internal/lsp/fake"
	"golang.org/x/tools/internal/lsp/protocol"
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
	runner.Run(t, exampleProgram, func(t *testing.T, env *Env) {
		// Deleting the 'n' at the end of Println should generate a single error
		// diagnostic.
		env.OpenFile("main.go")
		env.RegexpReplace("main.go", "Printl(n)", "")
		env.Await(
			env.DiagnosticAtRegexp("main.go", "Printl"),
			// Assert that this test has sent no error logs to the client. This is not
			// strictly necessary for testing this regression, but is included here
			// as an example of using the NoErrorLogs() expectation. Feel free to
			// delete.
			NoErrorLogs(),
		)
	})
}

const onlyMod = `
-- go.mod --
module mod.com

go 1.12
`

func TestMissingImportDiagsClearOnFirstFile(t *testing.T) {
	t.Parallel()
	runner.Run(t, onlyMod, func(t *testing.T, env *Env) {
		env.CreateBuffer("main.go", `package main

func m() {
	log.Println()
}
`)
		env.Await(
			env.DiagnosticAtRegexp("main.go", "log"),
		)
		env.SaveBuffer("main.go")
		env.Await(
			EmptyDiagnostics("main.go"),
		)
	})
}

const brokenFile = `package main

const Foo = "abc
`

func TestDiagnosticErrorInNewFile(t *testing.T) {
	runner.Run(t, brokenFile, func(t *testing.T, env *Env) {
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
	runner.Run(t, badPackage, func(t *testing.T, env *Env) {
		env.OpenFile("b.go")
		env.Await(env.DiagnosticAtRegexp("a.go", "a = 1"), env.DiagnosticAtRegexp("b.go", "a = 2"))

		// Fix the error by editing the const name in b.go to `b`.
		env.RegexpReplace("b.go", "(a) = 2", "b")
		env.Await(
			EmptyDiagnostics("a.go"),
			EmptyDiagnostics("b.go"),
		)
	})
}

func TestDiagnosticClearingOnDelete_Issue37049(t *testing.T) {
	runner.Run(t, badPackage, func(t *testing.T, env *Env) {
		env.OpenFile("a.go")
		env.Await(env.DiagnosticAtRegexp("a.go", "a = 1"), env.DiagnosticAtRegexp("b.go", "a = 2"))
		env.RemoveFileFromWorkspace("b.go")

		env.Await(EmptyDiagnostics("a.go"), EmptyDiagnostics("b.go"))
	})
}

func TestDiagnosticClearingOnClose(t *testing.T) {
	runner.Run(t, badPackage, func(t *testing.T, env *Env) {
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
	runner.Run(t, exampleProgram, func(t *testing.T, env *Env) {
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

const noMod = `
-- main.go --
package main

import "mod.com/bob"

func main() {
	bob.Hello()
}
-- bob/bob.go --
package bob

func Hello() {
	var x int
}
`

// TestNoMod confirms that gopls continues to work when a user adds a go.mod
// file to their workspace.
func TestNoMod(t *testing.T) {
	t.Run("manual", func(t *testing.T) {
		runner.Run(t, noMod, func(t *testing.T, env *Env) {
			env.Await(
				env.DiagnosticAtRegexp("main.go", `"mod.com/bob"`),
			)
			env.CreateBuffer("go.mod", `module mod.com

	go 1.12
`)
			env.SaveBuffer("go.mod")
			env.Await(
				EmptyDiagnostics("main.go"),
				env.DiagnosticAtRegexp("bob/bob.go", "x"),
			)
		})
	})
	t.Run("initialized", func(t *testing.T) {
		runner.Run(t, noMod, func(t *testing.T, env *Env) {
			env.Await(
				env.DiagnosticAtRegexp("main.go", `"mod.com/bob"`),
			)
			if err := env.W.RunGoCommand(env.Ctx, "mod", "init", "mod.com"); err != nil {
				t.Fatal(err)
			}
			env.Await(
				EmptyDiagnostics("main.go"),
				env.DiagnosticAtRegexp("bob/bob.go", "x"),
			)
		})
	})
}

const testPackage = `
-- go.mod --
module mod.com

go 1.12
-- lib.go --
package lib

func Hello(x string) {
	_ = x
}
-- lib_test.go --
package lib

import "testing"

type testStruct struct{
	name string
}

func TestHello(t *testing.T) {
	testStructs := []*testStruct{
		&testStruct{"hello"},
		&testStruct{"goodbye"},
	}
	for y := range testStructs {
		_ = y
	}
}
`

func Test_Issue38267(t *testing.T) {
	runner.Run(t, testPackage, func(t *testing.T, env *Env) {
		env.OpenFile("lib_test.go")
		env.Await(
			DiagnosticAt("lib_test.go", 10, 2),
			DiagnosticAt("lib_test.go", 11, 2),
		)
		env.OpenFile("lib.go")
		env.RegexpReplace("lib.go", "_ = x", "var y int")
		env.Await(
			env.DiagnosticAtRegexp("lib.go", "y int"),
			EmptyDiagnostics("lib_test.go"),
		)
	})
}

const packageChange = `
-- go.mod --
module fake
-- a.go --
package foo
func main() {}
`

func TestPackageChange(t *testing.T) {
	t.Skip("skipping due to golang.org/issues/32149")
	runner.Run(t, packageChange, func(t *testing.T, env *Env) {
		env.OpenFile("a.go")
		env.RegexpReplace("a.go", "foo", "foox")
		// TODO: there should be no error
		env.Await(EmptyDiagnostics("a.go"))
	})
}

const testPackageWithRequire = `
-- go.mod --
module mod.com

go 1.12

require (
	foo.test v1.2.3
)
-- print.go --
package lib

import (
	"fmt"

	"foo.test/bar"
)

func PrintAnswer() {
	fmt.Printf("answer: %s", bar.Answer)
}
`

const testPackageWithRequireProxy = `
-- foo.test@v1.2.3/go.mod --
module foo.test

go 1.12
-- foo.test@v1.2.3/bar/const.go --
package bar

const Answer = 42
`

func TestResolveDiagnosticWithDownload(t *testing.T) {
	runner.Run(t, testPackageWithRequire, func(t *testing.T, env *Env) {
		env.OpenFile("print.go")
		// Check that gopackages correctly loaded this dependency. We should get a
		// diagnostic for the wrong formatting type.
		// TODO: we should be able to easily also match the diagnostic message.
		env.Await(env.DiagnosticAtRegexp("print.go", "fmt.Printf"))
	}, WithProxy(testPackageWithRequireProxy))
}

func TestMissingDependency(t *testing.T) {
	runner.Run(t, testPackageWithRequire, func(t *testing.T, env *Env) {
		env.OpenFile("print.go")
		env.Await(LogMatching(protocol.Error, "initial workspace load failed"))
	})
}

func TestAdHocPackagesIssue_36951(t *testing.T) {
	const adHoc = `
-- b/b.go --
package b

func Hello() {
	var x int
}
`
	runner.Run(t, adHoc, func(t *testing.T, env *Env) {
		env.OpenFile("b/b.go")
		env.Await(env.DiagnosticAtRegexp("b/b.go", "x"))
	})
}

func TestNoGOPATHIssue_37984(t *testing.T) {
	const missingImport = `
-- main.go --
package main

func _() {
	fmt.Println("Hello World")
}
`
	runner.Run(t, missingImport, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.Await(env.DiagnosticAtRegexp("main.go", "fmt"))
		if err := env.E.OrganizeImports(env.Ctx, "main.go"); err == nil {
			t.Fatalf("organize imports should fail with an empty GOPATH")
		}
	}, WithEnv("GOPATH="))
}
