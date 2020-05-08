// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"fmt"
	"os"
	"testing"

	"golang.org/x/tools/internal/lsp"
	"golang.org/x/tools/internal/lsp/fake"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/tests"
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
	// This test is very basic: start with a clean Go program, make an error, and
	// get a diagnostic for that error. However, it also demonstrates how to
	// combine Expectations to await more complex state in the editor.
	runner.Run(t, exampleProgram, func(t *testing.T, env *Env) {
		// Deleting the 'n' at the end of Println should generate a single error
		// diagnostic.
		env.OpenFile("main.go")
		env.RegexpReplace("main.go", "Printl(n)", "")
		env.Await(
			// Once we have gotten diagnostics for the change above, we should
			// satisfy the DiagnosticAtRegexp assertion.
			OnceMet(
				CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidChange), 1),
				env.DiagnosticAtRegexp("main.go", "Printl"),
			),
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

// Tests golang/go#37978.
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

// TestNoMod confirms that gopls continues to work when a user adds a go.mod
// file to their workspace.
func TestNoMod(t *testing.T) {
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
			if err := env.Sandbox.RunGoCommand(env.Ctx, "mod", "init", "mod.com"); err != nil {
				t.Fatal(err)
			}
			env.Await(
				EmptyDiagnostics("main.go"),
				env.DiagnosticAtRegexp("bob/bob.go", "x"),
			)
		})
	})
}

// Tests golang/go#38267.
func TestIssue38267(t *testing.T) {
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

// Tests golang/go#38328.
func TestPackageChange_Issue38328(t *testing.T) {
	const packageChange = `
-- go.mod --
module fake
-- a.go --
package foo
func main() {}
`
	runner.Run(t, packageChange, func(t *testing.T, env *Env) {
		env.OpenFile("a.go")
		env.RegexpReplace("a.go", "foo", "foox")
		env.Await(
			// When the bug reported in #38328 was present, we didn't get erroneous
			// file diagnostics until after the didChange message generated by the
			// package renaming was fully processed. Therefore, in order for this
			// test to actually exercise the bug, we must wait until that work has
			// completed.
			OnceMet(
				CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidChange), 1),
				NoDiagnostics("a.go"),
			),
		)
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

// Tests golang/go#36951.
func TestAdHocPackages_Issue36951(t *testing.T) {
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

// Tests golang/go#37984.
func TestNoGOPATH_Issue37984(t *testing.T) {
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
		if err := env.Editor.OrganizeImports(env.Ctx, "main.go"); err == nil {
			t.Fatalf("organize imports should fail with an empty GOPATH")
		}
	}, WithEditorConfig(fake.EditorConfig{Env: []string{"GOPATH="}}))
}

// Tests golang/go#38669.
func TestEqualInEnv_Issue38669(t *testing.T) {
	const missingImport = `
-- go.mod --
module mod.com

-- main.go --
package main

var _ = x.X
-- x/x.go --
package x

var X = 0
`
	runner.Run(t, missingImport, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.OrganizeImports("main.go")
		env.Await(EmptyDiagnostics("main.go"))
	}, WithEditorConfig(fake.EditorConfig{Env: []string{"GOFLAGS=-tags=foo"}}))
}

// Tests golang/go#38467.
func TestNoSuggestedFixesForGeneratedFiles_Issue38467(t *testing.T) {
	const generated = `
-- go.mod --
module mod.com

-- main.go --
package main

// Code generated by generator.go. DO NOT EDIT.

func _() {
	for i, _ := range []string{} {
		_ = i
	}
}
`
	runner.Run(t, generated, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		original := env.ReadWorkspaceFile("main.go")
		metBy := env.Await(
			DiagnosticAt("main.go", 5, 8),
		)
		d, ok := metBy[0].(*protocol.PublishDiagnosticsParams)
		if !ok {
			t.Fatalf("unexpected met by result %v (%T)", metBy, metBy)
		}
		// Apply fixes and save the buffer.
		env.ApplyQuickFixes("main.go", d.Diagnostics)
		env.SaveBuffer("main.go")
		fixed := env.ReadWorkspaceFile("main.go")
		if original != fixed {
			t.Fatalf("generated file was changed by quick fixes:\n%s", tests.Diff(original, fixed))
		}
	})
}

// Expect a module/GOPATH error if there is an error in the file at startup.
// Tests golang/go#37279.
func TestShowMessage_Issue37279(t *testing.T) {
	const noModule = `
-- a.go --
package foo

func f() {
	fmt.Printl()
}
`
	runner.Run(t, noModule, func(t *testing.T, env *Env) {
		env.OpenFile("a.go")
		env.Await(env.DiagnosticAtRegexp("a.go", "fmt.Printl"), SomeShowMessage(""))
	})
}

// Expect no module/GOPATH error if there is no error in the file.
// Tests golang/go#37279.
func TestNoShowMessage_Issue37279(t *testing.T) {
	const noModule = `
-- a.go --
package foo

func f() {
}
`
	runner.Run(t, noModule, func(t *testing.T, env *Env) {
		env.OpenFile("a.go")
		env.Await(
			OnceMet(
				CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidOpen), 1),
				NoDiagnostics("a.go"),
			),
			EmptyShowMessage(""),
		)
		// introduce an error, expect no Show Message
		env.RegexpReplace("a.go", "func", "fun")
		env.Await(env.DiagnosticAtRegexp("a.go", "fun"), EmptyShowMessage(""))
	})
}

// Tests golang/go#38602.
func TestNonexistentFileDiagnostics_Issue38602(t *testing.T) {
	const collision = `
-- x/x.go --
package x

import "x/hello"

func Hello() {
	hello.HiThere()
}
-- x/main.go --
package main

func main() {
	fmt.Println("")
}
`
	runner.Run(t, collision, func(t *testing.T, env *Env) {
		env.OpenFile("x/main.go")
		env.Await(
			env.DiagnosticAtRegexp("x/main.go", "fmt.Println"),
		)
		env.OrganizeImports("x/main.go")
		// span.Parse misparses the error message when multiple packages are
		// defined in the same directory, creating a garbage filename.
		// Previously, we would send diagnostics for this nonexistent file.
		// This test checks that we don't send diagnostics for this file.
		dir, err := os.Getwd()
		if err != nil {
			t.Fatal(err)
		}
		badFile := fmt.Sprintf("%s/found packages main (main.go) and x (x.go) in %s/src/x", dir, env.Sandbox.GOPATH())
		env.Await(
			OnceMet(
				CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidChange), 1),
				EmptyDiagnostics("x/main.go"),
			),
			NoDiagnostics(badFile),
		)
	}, InGOPATH())
}
