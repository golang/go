// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package diagnostics

import (
	"context"
	"fmt"
	"os/exec"
	"testing"

	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/hooks"
	"golang.org/x/tools/gopls/internal/lsp"
	"golang.org/x/tools/gopls/internal/lsp/fake"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	. "golang.org/x/tools/gopls/internal/lsp/regtest"
	"golang.org/x/tools/internal/testenv"
)

func TestMain(m *testing.M) {
	bug.PanicOnBugs = true
	Main(m, hooks.Options)
}

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
	Run(t, exampleProgram, func(t *testing.T, env *Env) {
		// Deleting the 'n' at the end of Println should generate a single error
		// diagnostic.
		env.OpenFile("main.go")
		env.RegexpReplace("main.go", "Printl(n)", "")
		env.AfterChange(
			Diagnostics(env.AtRegexp("main.go", "Printl")),
			// Assert that this test has sent no error logs to the client. This is not
			// strictly necessary for testing this regression, but is included here
			// as an example of using the NoErrorLogs() expectation. Feel free to
			// delete.
			NoErrorLogs(),
		)
	})
}

func TestMissingImportDiagsClearOnFirstFile(t *testing.T) {
	const onlyMod = `
-- go.mod --
module mod.com

go 1.12
`
	Run(t, onlyMod, func(t *testing.T, env *Env) {
		env.CreateBuffer("main.go", `package main

func m() {
	log.Println()
}
`)
		env.AfterChange(Diagnostics(env.AtRegexp("main.go", "log")))
		env.SaveBuffer("main.go")
		env.AfterChange(NoDiagnostics(ForFile("main.go")))
	})
}

func TestDiagnosticErrorInNewFile(t *testing.T) {
	const brokenFile = `package main

const Foo = "abc
`
	Run(t, brokenFile, func(t *testing.T, env *Env) {
		env.CreateBuffer("broken.go", brokenFile)
		env.AfterChange(Diagnostics(env.AtRegexp("broken.go", "\"abc")))
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
	Run(t, badPackage, func(t *testing.T, env *Env) {
		env.OpenFile("b.go")
		env.AfterChange(
			Diagnostics(env.AtRegexp("a.go", "a = 1")),
			Diagnostics(env.AtRegexp("b.go", "a = 2")),
		)

		// Fix the error by editing the const name in b.go to `b`.
		env.RegexpReplace("b.go", "(a) = 2", "b")
		env.AfterChange(
			NoDiagnostics(ForFile("a.go")),
			NoDiagnostics(ForFile("b.go")),
		)
	})
}

func TestDiagnosticClearingOnDelete_Issue37049(t *testing.T) {
	Run(t, badPackage, func(t *testing.T, env *Env) {
		env.OpenFile("a.go")
		env.AfterChange(
			Diagnostics(env.AtRegexp("a.go", "a = 1")),
			Diagnostics(env.AtRegexp("b.go", "a = 2")),
		)
		env.RemoveWorkspaceFile("b.go")

		env.AfterChange(
			NoDiagnostics(ForFile("a.go")),
			NoDiagnostics(ForFile("b.go")),
		)
	})
}

func TestDiagnosticClearingOnClose(t *testing.T) {
	Run(t, badPackage, func(t *testing.T, env *Env) {
		env.CreateBuffer("c.go", `package consts

const a = 3`)
		env.AfterChange(
			Diagnostics(env.AtRegexp("a.go", "a = 1")),
			Diagnostics(env.AtRegexp("b.go", "a = 2")),
			Diagnostics(env.AtRegexp("c.go", "a = 3")),
		)
		env.CloseBuffer("c.go")
		env.AfterChange(
			Diagnostics(env.AtRegexp("a.go", "a = 1")),
			Diagnostics(env.AtRegexp("b.go", "a = 2")),
			NoDiagnostics(ForFile("c.go")),
		)
	})
}

// Tests golang/go#37978.
func TestIssue37978(t *testing.T) {
	Run(t, exampleProgram, func(t *testing.T, env *Env) {
		// Create a new workspace-level directory and empty file.
		env.CreateBuffer("c/c.go", "")

		// Write the file contents with a missing import.
		env.EditBuffer("c/c.go", protocol.TextEdit{
			NewText: `package c

const a = http.MethodGet
`,
		})
		env.AfterChange(
			Diagnostics(env.AtRegexp("c/c.go", "http.MethodGet")),
		)
		// Save file, which will organize imports, adding the expected import.
		// Expect the diagnostics to clear.
		env.SaveBuffer("c/c.go")
		env.AfterChange(
			NoDiagnostics(ForFile("c/c.go")),
		)
	})
}

// Tests golang/go#38878: good a.go, bad a_test.go, remove a_test.go but its errors remain
// If the file is open in the editor, this is working as intended
// If the file is not open in the editor, the errors go away
const test38878 = `
-- go.mod --
module foo

go 1.12
-- a.go --
package x

// import "fmt"

func f() {}

-- a_test.go --
package x

import "testing"

func TestA(t *testing.T) {
	f(3)
}
`

// Tests golang/go#38878: deleting a test file should clear its errors, and
// not break the workspace.
func TestDeleteTestVariant(t *testing.T) {
	Run(t, test38878, func(t *testing.T, env *Env) {
		env.AfterChange(Diagnostics(env.AtRegexp("a_test.go", `f\((3)\)`)))
		env.RemoveWorkspaceFile("a_test.go")
		env.AfterChange(NoDiagnostics(ForFile("a_test.go")))

		// Make sure the test variant has been removed from the workspace by
		// triggering a metadata load.
		env.OpenFile("a.go")
		env.RegexpReplace("a.go", `// import`, "import")
		env.AfterChange(Diagnostics(env.AtRegexp("a.go", `"fmt"`)))
	})
}

// Tests golang/go#38878: deleting a test file on disk while it's still open
// should not clear its errors.
func TestDeleteTestVariant_DiskOnly(t *testing.T) {
	Run(t, test38878, func(t *testing.T, env *Env) {
		env.OpenFile("a_test.go")
		env.AfterChange(Diagnostics(AtPosition("a_test.go", 5, 3)))
		env.Sandbox.Workdir.RemoveFile(context.Background(), "a_test.go")
		env.AfterChange(Diagnostics(AtPosition("a_test.go", 5, 3)))
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
		Run(t, noMod, func(t *testing.T, env *Env) {
			env.OnceMet(
				InitialWorkspaceLoad,
				Diagnostics(env.AtRegexp("main.go", `"mod.com/bob"`)),
			)
			env.CreateBuffer("go.mod", `module mod.com

	go 1.12
`)
			env.SaveBuffer("go.mod")
			var d protocol.PublishDiagnosticsParams
			env.AfterChange(
				NoDiagnostics(ForFile("main.go")),
				Diagnostics(env.AtRegexp("bob/bob.go", "x")),
				ReadDiagnostics("bob/bob.go", &d),
			)
			if len(d.Diagnostics) != 1 {
				t.Fatalf("expected 1 diagnostic, got %v", len(d.Diagnostics))
			}
		})
	})
	t.Run("initialized", func(t *testing.T) {
		Run(t, noMod, func(t *testing.T, env *Env) {
			env.OnceMet(
				InitialWorkspaceLoad,
				Diagnostics(env.AtRegexp("main.go", `"mod.com/bob"`)),
			)
			env.RunGoCommand("mod", "init", "mod.com")
			env.AfterChange(
				NoDiagnostics(ForFile("main.go")),
				Diagnostics(env.AtRegexp("bob/bob.go", "x")),
			)
		})
	})

	t.Run("without workspace module", func(t *testing.T) {
		WithOptions(
			Modes(Default),
		).Run(t, noMod, func(t *testing.T, env *Env) {
			env.OnceMet(
				InitialWorkspaceLoad,
				Diagnostics(env.AtRegexp("main.go", `"mod.com/bob"`)),
			)
			if err := env.Sandbox.RunGoCommand(env.Ctx, "", "mod", []string{"init", "mod.com"}, nil, true); err != nil {
				t.Fatal(err)
			}
			env.AfterChange(
				NoDiagnostics(ForFile("main.go")),
				Diagnostics(env.AtRegexp("bob/bob.go", "x")),
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

	Run(t, testPackage, func(t *testing.T, env *Env) {
		env.OpenFile("lib_test.go")
		env.AfterChange(
			Diagnostics(AtPosition("lib_test.go", 10, 2)),
			Diagnostics(AtPosition("lib_test.go", 11, 2)),
		)
		env.OpenFile("lib.go")
		env.RegexpReplace("lib.go", "_ = x", "var y int")
		env.AfterChange(
			Diagnostics(env.AtRegexp("lib.go", "y int")),
			NoDiagnostics(ForFile("lib_test.go")),
		)
	})
}

// Tests golang/go#38328.
func TestPackageChange_Issue38328(t *testing.T) {
	const packageChange = `
-- go.mod --
module fake

go 1.12
-- a.go --
package foo
func main() {}
`
	Run(t, packageChange, func(t *testing.T, env *Env) {
		env.OpenFile("a.go")
		env.RegexpReplace("a.go", "foo", "foox")
		env.AfterChange(
			NoDiagnostics(ForFile("a.go")),
		)
	})
}

const testPackageWithRequire = `
-- go.mod --
module mod.com

go 1.12

require foo.test v1.2.3
-- go.sum --
foo.test v1.2.3 h1:TMA+lyd1ck0TqjSFpNe4T6cf/K6TYkoHwOOcMBMjaEw=
foo.test v1.2.3/go.mod h1:Ij3kyLIe5lzjycjh13NL8I2gX0quZuTdW0MnmlwGBL4=
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
	WithOptions(
		ProxyFiles(testPackageWithRequireProxy),
	).Run(t, testPackageWithRequire, func(t *testing.T, env *Env) {
		env.OpenFile("print.go")
		// Check that gopackages correctly loaded this dependency. We should get a
		// diagnostic for the wrong formatting type.
		env.AfterChange(
			Diagnostics(
				env.AtRegexp("print.go", "fmt.Printf"),
				WithMessage("wrong type int"),
			),
		)
	})
}

func TestMissingDependency(t *testing.T) {
	Run(t, testPackageWithRequire, func(t *testing.T, env *Env) {
		env.OpenFile("print.go")
		env.Await(
			// Log messages are asynchronous to other events on the LSP stream, so we
			// can't use OnceMet or AfterChange here.
			LogMatching(protocol.Error, "initial workspace load failed", 1, false),
		)
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
	Run(t, adHoc, func(t *testing.T, env *Env) {
		env.OpenFile("b/b.go")
		env.AfterChange(
			Diagnostics(env.AtRegexp("b/b.go", "x")),
		)
	})
}

// Tests golang/go#37984: GOPATH should be read from the go command.
func TestNoGOPATH_Issue37984(t *testing.T) {
	const files = `
-- main.go --
package main

func _() {
	fmt.Println("Hello World")
}
`
	WithOptions(
		EnvVars{
			"GOPATH":      "",
			"GO111MODULE": "off",
		},
	).Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.AfterChange(Diagnostics(env.AtRegexp("main.go", "fmt")))
		env.SaveBuffer("main.go")
		env.AfterChange(NoDiagnostics(ForFile("main.go")))
	})
}

// Tests golang/go#38669.
func TestEqualInEnv_Issue38669(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

var _ = x.X
-- x/x.go --
package x

var X = 0
`
	WithOptions(
		EnvVars{"GOFLAGS": "-tags=foo"},
	).Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.OrganizeImports("main.go")
		env.AfterChange(NoDiagnostics(ForFile("main.go")))
	})
}

// Tests golang/go#38467.
func TestNoSuggestedFixesForGeneratedFiles_Issue38467(t *testing.T) {
	const generated = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

// Code generated by generator.go. DO NOT EDIT.

func _() {
	for i, _ := range []string{} {
		_ = i
	}
}
`
	Run(t, generated, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		var d protocol.PublishDiagnosticsParams
		env.AfterChange(
			Diagnostics(AtPosition("main.go", 5, 8)),
			ReadDiagnostics("main.go", &d),
		)
		if fixes := env.GetQuickFixes("main.go", d.Diagnostics); len(fixes) != 0 {
			t.Errorf("got quick fixes %v, wanted none", fixes)
		}
	})
}

// Expect a module/GOPATH error if there is an error in the file at startup.
// Tests golang/go#37279.
func TestBrokenWorkspace_OutsideModule(t *testing.T) {
	const noModule = `
-- a.go --
package foo

import "mod.com/hello"

func f() {
	hello.Goodbye()
}
`
	Run(t, noModule, func(t *testing.T, env *Env) {
		env.OpenFile("a.go")
		env.AfterChange(
			// Expect the adHocPackagesWarning.
			OutstandingWork(lsp.WorkspaceLoadFailure, "outside of a module"),
		)
		// Deleting the import dismisses the warning.
		env.RegexpReplace("a.go", `import "mod.com/hello"`, "")
		env.AfterChange(
			NoOutstandingWork(IgnoreTelemetryPromptWork),
		)
	})
}

func TestNonGoFolder(t *testing.T) {
	const files = `
-- hello.txt --
hi mom
`
	for _, go111module := range []string{"on", "off", ""} {
		t.Run(fmt.Sprintf("GO111MODULE_%v", go111module), func(t *testing.T) {
			WithOptions(
				EnvVars{"GO111MODULE": go111module},
			).Run(t, files, func(t *testing.T, env *Env) {
				env.OnceMet(
					InitialWorkspaceLoad,
					NoOutstandingWork(IgnoreTelemetryPromptWork),
				)
			})
		})
	}
}

// Tests the repro case from golang/go#38602. Diagnostics are now handled properly,
// which blocks type checking.
func TestConflictingMainPackageErrors(t *testing.T) {
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
	WithOptions(
		InGOPATH(),
		EnvVars{"GO111MODULE": "off"},
	).Run(t, collision, func(t *testing.T, env *Env) {
		env.OpenFile("x/x.go")
		env.AfterChange(
			Diagnostics(env.AtRegexp("x/x.go", `^`), WithMessage("found packages main (main.go) and x (x.go)")),
			Diagnostics(env.AtRegexp("x/main.go", `^`), WithMessage("found packages main (main.go) and x (x.go)")),
		)

		// We don't recover cleanly from the errors without good overlay support.
		if testenv.Go1Point() >= 16 {
			env.RegexpReplace("x/x.go", `package x`, `package main`)
			env.AfterChange(
				Diagnostics(env.AtRegexp("x/main.go", `fmt`)),
			)
		}
	})
}

const ardanLabsProxy = `
-- github.com/ardanlabs/conf@v1.2.3/go.mod --
module github.com/ardanlabs/conf

go 1.12
-- github.com/ardanlabs/conf@v1.2.3/conf.go --
package conf

var ErrHelpWanted error
`

// Test for golang/go#38211.
func Test_Issue38211(t *testing.T) {
	const ardanLabs = `
-- go.mod --
module mod.com

go 1.14
-- main.go --
package main

import "github.com/ardanlabs/conf"

func main() {
	_ = conf.ErrHelpWanted
}
`
	WithOptions(
		ProxyFiles(ardanLabsProxy),
	).Run(t, ardanLabs, func(t *testing.T, env *Env) {
		// Expect a diagnostic with a suggested fix to add
		// "github.com/ardanlabs/conf" to the go.mod file.
		env.OpenFile("go.mod")
		env.OpenFile("main.go")
		var d protocol.PublishDiagnosticsParams
		env.AfterChange(
			Diagnostics(env.AtRegexp("main.go", `"github.com/ardanlabs/conf"`)),
			ReadDiagnostics("main.go", &d),
		)
		env.ApplyQuickFixes("main.go", d.Diagnostics)
		env.SaveBuffer("go.mod")
		env.AfterChange(
			NoDiagnostics(ForFile("main.go")),
		)
		// Comment out the line that depends on conf and expect a
		// diagnostic and a fix to remove the import.
		env.RegexpReplace("main.go", "_ = conf.ErrHelpWanted", "//_ = conf.ErrHelpWanted")
		env.AfterChange(
			Diagnostics(env.AtRegexp("main.go", `"github.com/ardanlabs/conf"`)),
		)
		env.SaveBuffer("main.go")
		// Expect a diagnostic and fix to remove the dependency in the go.mod.
		env.AfterChange(
			NoDiagnostics(ForFile("main.go")),
			Diagnostics(env.AtRegexp("go.mod", "require github.com/ardanlabs/conf"), WithMessage("not used in this module")),
			ReadDiagnostics("go.mod", &d),
		)
		env.ApplyQuickFixes("go.mod", d.Diagnostics)
		env.SaveBuffer("go.mod")
		env.AfterChange(
			NoDiagnostics(ForFile("go.mod")),
		)
		// Uncomment the lines and expect a new diagnostic for the import.
		env.RegexpReplace("main.go", "//_ = conf.ErrHelpWanted", "_ = conf.ErrHelpWanted")
		env.SaveBuffer("main.go")
		env.AfterChange(
			Diagnostics(env.AtRegexp("main.go", `"github.com/ardanlabs/conf"`)),
		)
	})
}

// Test for golang/go#38207.
func TestNewModule_Issue38207(t *testing.T) {
	const emptyFile = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
`
	WithOptions(
		ProxyFiles(ardanLabsProxy),
	).Run(t, emptyFile, func(t *testing.T, env *Env) {
		env.CreateBuffer("main.go", `package main

import "github.com/ardanlabs/conf"

func main() {
	_ = conf.ErrHelpWanted
}
`)
		env.SaveBuffer("main.go")
		var d protocol.PublishDiagnosticsParams
		env.AfterChange(
			Diagnostics(env.AtRegexp("main.go", `"github.com/ardanlabs/conf"`), WithMessage("no required module")),
			ReadDiagnostics("main.go", &d),
		)
		env.ApplyQuickFixes("main.go", d.Diagnostics)
		env.AfterChange(
			NoDiagnostics(ForFile("main.go")),
		)
	})
}

// Test for golang/go#36960.
func TestNewFileBadImports_Issue36960(t *testing.T) {
	const simplePackage = `
-- go.mod --
module mod.com

go 1.14
-- a/a1.go --
package a

import "fmt"

func _() {
	fmt.Println("hi")
}
`
	Run(t, simplePackage, func(t *testing.T, env *Env) {
		env.OpenFile("a/a1.go")
		env.CreateBuffer("a/a2.go", ``)
		env.SaveBufferWithoutActions("a/a2.go")
		env.AfterChange(
			NoDiagnostics(ForFile("a/a1.go")),
		)
		env.EditBuffer("a/a2.go", fake.NewEdit(0, 0, 0, 0, `package a`))
		env.AfterChange(
			NoDiagnostics(ForFile("a/a1.go")),
		)
	})
}

// This test tries to replicate the workflow of a user creating a new x test.
// It also tests golang/go#39315.
func TestManuallyCreatingXTest(t *testing.T) {
	// Create a package that already has a test variant (in-package test).
	const testVariant = `
-- go.mod --
module mod.com

go 1.15
-- hello/hello.go --
package hello

func Hello() {
	var x int
}
-- hello/hello_test.go --
package hello

import "testing"

func TestHello(t *testing.T) {
	var x int
	Hello()
}
`
	Run(t, testVariant, func(t *testing.T, env *Env) {
		// Open the file, triggering the workspace load.
		// There are errors in the code to ensure all is working as expected.
		env.OpenFile("hello/hello.go")
		env.AfterChange(
			Diagnostics(env.AtRegexp("hello/hello.go", "x")),
			Diagnostics(env.AtRegexp("hello/hello_test.go", "x")),
		)

		// Create an empty file with the intention of making it an x test.
		// This resembles a typical flow in an editor like VS Code, in which
		// a user would create an empty file and add content, saving
		// intermittently.
		// TODO(rstambler): There might be more edge cases here, as file
		// content can be added incrementally.
		env.CreateBuffer("hello/hello_x_test.go", ``)

		// Save the empty file (no actions since formatting will fail).
		env.SaveBufferWithoutActions("hello/hello_x_test.go")

		// Add the content. The missing import is for the package under test.
		env.EditBuffer("hello/hello_x_test.go", fake.NewEdit(0, 0, 0, 0, `package hello_test

import (
	"testing"
)

func TestHello(t *testing.T) {
	hello.Hello()
}
`))
		// Expect a diagnostic for the missing import. Save, which should
		// trigger import organization. The diagnostic should clear.
		env.AfterChange(
			Diagnostics(env.AtRegexp("hello/hello_x_test.go", "hello.Hello")),
		)
		env.SaveBuffer("hello/hello_x_test.go")
		env.AfterChange(
			NoDiagnostics(ForFile("hello/hello_x_test.go")),
		)
	})
}

// Reproduce golang/go#40690.
func TestCreateOnlyXTest(t *testing.T) {
	const mod = `
-- go.mod --
module mod.com

go 1.12
-- foo/foo.go --
package foo
-- foo/bar_test.go --
`
	Run(t, mod, func(t *testing.T, env *Env) {
		env.OpenFile("foo/bar_test.go")
		env.EditBuffer("foo/bar_test.go", fake.NewEdit(0, 0, 0, 0, "package foo"))
		env.Await(env.DoneWithChange())
		env.RegexpReplace("foo/bar_test.go", "package foo", `package foo_test

import "testing"

func TestX(t *testing.T) {
	var x int
}
`)
		env.AfterChange(
			Diagnostics(env.AtRegexp("foo/bar_test.go", "x")),
		)
	})
}

func TestChangePackageName(t *testing.T) {
	const mod = `
-- go.mod --
module mod.com

go 1.12
-- foo/foo.go --
package foo
-- foo/bar_test.go --
package foo_
`
	Run(t, mod, func(t *testing.T, env *Env) {
		env.OpenFile("foo/bar_test.go")
		env.AfterChange()
		env.RegexpReplace("foo/bar_test.go", "package foo_", "package foo_test")
		env.AfterChange(
			NoDiagnostics(ForFile("foo/bar_test.go")),
			NoDiagnostics(ForFile("foo/foo.go")),
		)
	})
}

func TestIgnoredFiles(t *testing.T) {
	const ws = `
-- go.mod --
module mod.com

go 1.12
-- _foo/x.go --
package x

var _ = foo.Bar
`
	Run(t, ws, func(t *testing.T, env *Env) {
		env.OpenFile("_foo/x.go")
		env.AfterChange(
			NoDiagnostics(ForFile("_foo/x.go")),
		)
	})
}

// Partially reproduces golang/go#38977, moving a file between packages.
// It also gets hit by some go command bug fixed in 1.15, but we don't
// care about that so much here.
func TestDeletePackage(t *testing.T) {
	const ws = `
-- go.mod --
module mod.com

go 1.15
-- a/a.go --
package a

const A = 1

-- b/b.go --
package b

import "mod.com/a"

const B = a.A

-- c/c.go --
package c

import "mod.com/a"

const C = a.A
`
	Run(t, ws, func(t *testing.T, env *Env) {
		env.OpenFile("b/b.go")
		env.Await(env.DoneWithOpen())
		// Delete c/c.go, the only file in package c.
		env.RemoveWorkspaceFile("c/c.go")

		// We should still get diagnostics for files that exist.
		env.RegexpReplace("b/b.go", `a.A`, "a.Nonexistant")
		env.AfterChange(
			Diagnostics(env.AtRegexp("b/b.go", `Nonexistant`)),
		)
	})
}

// This is a copy of the scenario_default/quickfix_empty_files.txt test from
// govim. Reproduces golang/go#39646.
func TestQuickFixEmptyFiles(t *testing.T) {
	const mod = `
-- go.mod --
module mod.com

go 1.12
`
	// To fully recreate the govim tests, we create files by inserting
	// a newline, adding to the file, and then deleting the newline.
	// Wait for each event to process to avoid cancellations and force
	// package loads.
	writeGoVim := func(env *Env, name, content string) {
		env.WriteWorkspaceFile(name, "")
		env.Await(env.DoneWithChangeWatchedFiles())

		env.CreateBuffer(name, "\n")
		env.Await(env.DoneWithOpen())

		env.EditBuffer(name, fake.NewEdit(1, 0, 1, 0, content))
		env.Await(env.DoneWithChange())

		env.EditBuffer(name, fake.NewEdit(0, 0, 1, 0, ""))
		env.Await(env.DoneWithChange())
	}

	const p = `package p; func DoIt(s string) {};`
	const main = `package main

import "mod.com/p"

func main() {
	p.DoIt(5)
}
`
	// A simple version of the test that reproduces most of the problems it
	// exposes.
	t.Run("short", func(t *testing.T) {
		Run(t, mod, func(t *testing.T, env *Env) {
			writeGoVim(env, "p/p.go", p)
			writeGoVim(env, "main.go", main)
			env.AfterChange(
				Diagnostics(env.AtRegexp("main.go", "5")),
			)
		})
	})

	// A full version that replicates the whole flow of the test.
	t.Run("full", func(t *testing.T) {
		Run(t, mod, func(t *testing.T, env *Env) {
			writeGoVim(env, "p/p.go", p)
			writeGoVim(env, "main.go", main)
			writeGoVim(env, "p/p_test.go", `package p

import "testing"

func TestDoIt(t *testing.T) {
	DoIt(5)
}
`)
			writeGoVim(env, "p/x_test.go", `package p_test

import (
	"testing"

	"mod.com/p"
)

func TestDoIt(t *testing.T) {
	p.DoIt(5)
}
`)
			env.AfterChange(
				Diagnostics(env.AtRegexp("main.go", "5")),
				Diagnostics(env.AtRegexp("p/p_test.go", "5")),
				Diagnostics(env.AtRegexp("p/x_test.go", "5")),
			)
			env.RegexpReplace("p/p.go", "s string", "i int")
			env.AfterChange(
				NoDiagnostics(ForFile("main.go")),
				NoDiagnostics(ForFile("p/p_test.go")),
				NoDiagnostics(ForFile("p/x_test.go")),
			)
		})
	})
}

func TestSingleFile(t *testing.T) {
	const mod = `
-- go.mod --
module mod.com

go 1.13
-- a/a.go --
package a

func _() {
	var x int
}
`
	WithOptions(
		// Empty workspace folders.
		WorkspaceFolders(),
	).Run(t, mod, func(t *testing.T, env *Env) {
		env.OpenFile("a/a.go")
		env.AfterChange(
			Diagnostics(env.AtRegexp("a/a.go", "x")),
		)
	})
}

// Reproduces the case described in
// https://github.com/golang/go/issues/39296#issuecomment-652058883.
func TestPkgm(t *testing.T) {
	const basic = `
-- go.mod --
module mod.com

go 1.15
-- foo/foo.go --
package foo

import "fmt"

func Foo() {
	fmt.Println("")
}
`
	Run(t, basic, func(t *testing.T, env *Env) {
		env.WriteWorkspaceFile("foo/foo_test.go", `package main

func main() {

}`)
		env.OpenFile("foo/foo_test.go")
		env.RegexpReplace("foo/foo_test.go", `package main`, `package foo`)
		env.AfterChange(NoDiagnostics(ForFile("foo/foo.go")))
	})
}

func TestClosingBuffer(t *testing.T) {
	const basic = `
-- go.mod --
module mod.com

go 1.14
-- main.go --
package main

func main() {}
`
	Run(t, basic, func(t *testing.T, env *Env) {
		env.Editor.CreateBuffer(env.Ctx, "foo.go", `package main`)
		env.AfterChange()
		env.CloseBuffer("foo.go")
		env.AfterChange(NoLogMatching(protocol.Info, "packages=0"))
	})
}

// Reproduces golang/go#38424.
func TestCutAndPaste(t *testing.T) {
	const basic = `
-- go.mod --
module mod.com

go 1.14
-- main2.go --
package main
`
	Run(t, basic, func(t *testing.T, env *Env) {
		env.CreateBuffer("main.go", "")
		env.Await(env.DoneWithOpen())

		env.SaveBufferWithoutActions("main.go")
		env.Await(env.DoneWithSave(), env.DoneWithChangeWatchedFiles())

		env.EditBuffer("main.go", fake.NewEdit(0, 0, 0, 0, `package main

func main() {
}
`))
		env.Await(env.DoneWithChange())

		env.SaveBuffer("main.go")
		env.Await(env.DoneWithSave(), env.DoneWithChangeWatchedFiles())

		env.EditBuffer("main.go", fake.NewEdit(0, 0, 4, 0, ""))
		env.Await(env.DoneWithChange())

		env.EditBuffer("main.go", fake.NewEdit(0, 0, 0, 0, `package main

func main() {
	var x int
}
`))
		env.AfterChange(
			Diagnostics(env.AtRegexp("main.go", "x")),
		)
	})
}

// Reproduces golang/go#39763.
func TestInvalidPackageName(t *testing.T) {
	const pkgDefault = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package default

func main() {}
`
	Run(t, pkgDefault, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.AfterChange(
			Diagnostics(
				env.AtRegexp("main.go", "default"),
				WithMessage("expected 'IDENT'"),
			),
		)
	})
}

// This tests the functionality of the "limitWorkspaceScope"
func TestLimitWorkspaceScope(t *testing.T) {
	const mod = `
-- go.mod --
module mod.com

go 1.12
-- a/main.go --
package main

func main() {}
-- main.go --
package main

func main() {
	var x int
}
`
	WithOptions(
		WorkspaceFolders("a"),
	).Run(t, mod, func(t *testing.T, env *Env) {
		env.OpenFile("a/main.go")
		env.AfterChange(
			Diagnostics(env.AtRegexp("main.go", "x")),
		)
	})
	WithOptions(
		WorkspaceFolders("a"),
		Settings{"expandWorkspaceToModule": false},
	).Run(t, mod, func(t *testing.T, env *Env) {
		env.OpenFile("a/main.go")
		env.AfterChange(
			NoDiagnostics(ForFile("main.go")),
		)
	})
}

func TestSimplifyCompositeLitDiagnostic(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

import "fmt"

type t struct {
	msg string
}

func main() {
	x := []t{t{"msg"}}
	fmt.Println(x)
}
`

	WithOptions(
		Settings{"staticcheck": true},
	).Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		var d protocol.PublishDiagnosticsParams
		env.AfterChange(
			Diagnostics(env.AtRegexp("main.go", `t{"msg"}`), WithMessage("redundant type")),
			ReadDiagnostics("main.go", &d),
		)
		if tags := d.Diagnostics[0].Tags; len(tags) == 0 || tags[0] != protocol.Unnecessary {
			t.Errorf("wanted Unnecessary tag on diagnostic, got %v", tags)
		}
		env.ApplyQuickFixes("main.go", d.Diagnostics)
		env.AfterChange(NoDiagnostics(ForFile("main.go")))
	})
}

// Test some secondary diagnostics
func TestSecondaryDiagnostics(t *testing.T) {
	const dir = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main
func main() {
	panic("not here")
}
-- other.go --
package main
func main() {}
`
	Run(t, dir, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.OpenFile("other.go")
		var mainDiags, otherDiags protocol.PublishDiagnosticsParams
		env.AfterChange(
			ReadDiagnostics("main.go", &mainDiags),
			ReadDiagnostics("other.go", &otherDiags),
		)
		if len(mainDiags.Diagnostics) != 1 {
			t.Fatalf("main.go, got %d diagnostics, expected 1", len(mainDiags.Diagnostics))
		}
		keep := mainDiags.Diagnostics[0]
		if len(otherDiags.Diagnostics) != 1 {
			t.Fatalf("other.go: got %d diagnostics, expected 1", len(otherDiags.Diagnostics))
		}
		if len(otherDiags.Diagnostics[0].RelatedInformation) != 1 {
			t.Fatalf("got %d RelatedInformations, expected 1", len(otherDiags.Diagnostics[0].RelatedInformation))
		}
		// check that the RelatedInformation matches the error from main.go
		c := otherDiags.Diagnostics[0].RelatedInformation[0]
		if c.Location.Range != keep.Range {
			t.Errorf("locations don't match. Got %v expected %v", c.Location.Range, keep.Range)
		}
	})
}

func TestOrphanedFiles(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.12
-- a/a.go --
package a

func main() {
	var x int
}
-- a/a_exclude.go --
// +build exclude

package a

func _() {
	var x int
}
`
	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("a/a.go")
		env.AfterChange(
			Diagnostics(env.AtRegexp("a/a.go", "x")),
		)
		env.OpenFile("a/a_exclude.go")

		loadOnce := LogMatching(protocol.Info, "query=.*file=.*a_exclude.go", 1, false)

		// can't use OnceMet or AfterChange as logs are async
		env.Await(loadOnce)
		// ...but ensure that the change has been fully processed before editing.
		// Otherwise, there may be a race where the snapshot is cloned before all
		// state changes resulting from the load have been processed
		// (golang/go#61521).
		env.AfterChange()

		// Check that orphaned files are not reloaded, by making a change in
		// a.go file and confirming that the workspace diagnosis did not reload
		// a_exclude.go.
		//
		// This is racy (but fails open) because logs are asynchronous to other LSP
		// operations. There's a chance gopls _did_ log, and we just haven't seen
		// it yet.
		env.RegexpReplace("a/a.go", "package a", "package a // arbitrary comment")
		env.AfterChange(loadOnce)
	})
}

func TestEnableAllExperiments(t *testing.T) {
	// Before the oldest supported Go version, gopls sends a warning to upgrade
	// Go, which fails the expectation below.
	testenv.NeedsGo1Point(t, lsp.OldestSupportedGoVersion())

	const mod = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

import "bytes"

func b(c bytes.Buffer) {
	_ = 1
}
`
	WithOptions(
		Settings{"allExperiments": true},
	).Run(t, mod, func(t *testing.T, env *Env) {
		// Confirm that the setting doesn't cause any warnings.
		env.OnceMet(
			InitialWorkspaceLoad,
			NoShownMessage(""), // empty substring to match any message
		)
	})
}

func TestSwig(t *testing.T) {
	// This is fixed in Go 1.17, but not earlier.
	testenv.NeedsGo1Point(t, 17)

	if _, err := exec.LookPath("swig"); err != nil {
		t.Skip("skipping test: swig not available")
	}
	if _, err := exec.LookPath("g++"); err != nil {
		t.Skip("skipping test: g++ not available")
	}

	const mod = `
-- go.mod --
module mod.com

go 1.12
-- pkg/simple/export_swig.go --
package simple

func ExportSimple(x, y int) int {
	return Gcd(x, y)
}
-- pkg/simple/simple.swigcxx --
%module simple

%inline %{
extern int gcd(int x, int y)
{
  int g;
  g = y;
  while (x > 0) {
    g = x;
    x = y % x;
    y = g;
  }
  return g;
}
%}
-- main.go --
package a

func main() {
	var x int
}
`
	Run(t, mod, func(t *testing.T, env *Env) {
		env.OnceMet(
			InitialWorkspaceLoad,
			NoDiagnostics(WithMessage("illegal character U+0023 '#'")),
		)
	})
}

// When foo_test.go is opened, gopls will object to the borked package name.
// This test asserts that when the package name is fixed, gopls will soon after
// have no more complaints about it.
// https://github.com/golang/go/issues/41061
func TestRenamePackage(t *testing.T) {
	const proxy = `
-- example.com@v1.2.3/go.mod --
module example.com

go 1.12
-- example.com@v1.2.3/blah/blah.go --
package blah

const Name = "Blah"
-- random.org@v1.2.3/go.mod --
module random.org

go 1.12
-- random.org@v1.2.3/blah/blah.go --
package hello

const Name = "Hello"
`

	const contents = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

import "example.com/blah"

func main() {
	blah.Hello()
}
-- bob.go --
package main
-- foo/foo.go --
package foo
-- foo/foo_test.go --
package foo_
`

	WithOptions(
		ProxyFiles(proxy),
		InGOPATH(),
		EnvVars{"GO111MODULE": "off"},
	).Run(t, contents, func(t *testing.T, env *Env) {
		// Simulate typing character by character.
		env.OpenFile("foo/foo_test.go")
		env.Await(env.DoneWithOpen())
		env.RegexpReplace("foo/foo_test.go", "_", "_t")
		env.Await(env.DoneWithChange())
		env.RegexpReplace("foo/foo_test.go", "_t", "_test")
		env.AfterChange(
			NoDiagnostics(ForFile("foo/foo_test.go")),
			NoOutstandingWork(IgnoreTelemetryPromptWork),
		)
	})
}

// TestProgressBarErrors confirms that critical workspace load errors are shown
// and updated via progress reports.
func TestProgressBarErrors(t *testing.T) {
	const pkg = `
-- go.mod --
modul mod.com

go 1.12
-- main.go --
package main
`
	Run(t, pkg, func(t *testing.T, env *Env) {
		env.OpenFile("go.mod")
		env.AfterChange(
			OutstandingWork(lsp.WorkspaceLoadFailure, "unknown directive"),
		)
		env.EditBuffer("go.mod", fake.NewEdit(0, 0, 3, 0, `module mod.com

go 1.hello
`))
		// As of golang/go#42529, go.mod changes do not reload the workspace until
		// they are saved.
		env.SaveBufferWithoutActions("go.mod")
		env.AfterChange(
			OutstandingWork(lsp.WorkspaceLoadFailure, "invalid go version"),
		)
		env.RegexpReplace("go.mod", "go 1.hello", "go 1.12")
		env.SaveBufferWithoutActions("go.mod")
		env.AfterChange(
			NoOutstandingWork(IgnoreTelemetryPromptWork),
		)
	})
}

func TestDeleteDirectory(t *testing.T) {
	const mod = `
-- bob/bob.go --
package bob

func Hello() {
	var x int
}
-- go.mod --
module mod.com
-- cmd/main.go --
package main

import "mod.com/bob"

func main() {
	bob.Hello()
}
`
	WithOptions(
		Settings{
			// Now that we don't watch subdirs by default (except for VS Code),
			// we must explicitly ask gopls to requests subdir watch patterns.
			"subdirWatchPatterns": "on",
		},
	).Run(t, mod, func(t *testing.T, env *Env) {
		env.OnceMet(
			InitialWorkspaceLoad,
			FileWatchMatching("bob"),
		)
		env.RemoveWorkspaceFile("bob")
		env.AfterChange(
			Diagnostics(env.AtRegexp("cmd/main.go", `"mod.com/bob"`)),
			NoDiagnostics(ForFile("bob/bob.go")),
			NoFileWatchMatching("bob"),
		)
	})
}

// Confirms that circular imports are tested and reported.
func TestCircularImports(t *testing.T) {
	const mod = `
-- go.mod --
module mod.com

go 1.12
-- self/self.go --
package self

import _ "mod.com/self"
func Hello() {}
-- double/a/a.go --
package a

import _ "mod.com/double/b"
-- double/b/b.go --
package b

import _ "mod.com/double/a"
-- triple/a/a.go --
package a

import _ "mod.com/triple/b"
-- triple/b/b.go --
package b

import _ "mod.com/triple/c"
-- triple/c/c.go --
package c

import _ "mod.com/triple/a"
`
	Run(t, mod, func(t *testing.T, env *Env) {
		env.OnceMet(
			InitialWorkspaceLoad,
			Diagnostics(env.AtRegexp("self/self.go", `_ "mod.com/self"`), WithMessage("import cycle not allowed")),
			Diagnostics(env.AtRegexp("double/a/a.go", `_ "mod.com/double/b"`), WithMessage("import cycle not allowed")),
			Diagnostics(env.AtRegexp("triple/a/a.go", `_ "mod.com/triple/b"`), WithMessage("import cycle not allowed")),
		)
	})
}

// Tests golang/go#46667: deleting a problematic import path should resolve
// import cycle errors.
func TestResolveImportCycle(t *testing.T) {
	const mod = `
-- go.mod --
module mod.test

go 1.16
-- a/a.go --
package a

import "mod.test/b"

const A = b.A
const B = 2
-- b/b.go --
package b

import "mod.test/a"

const A = 1
const B = a.B
	`
	Run(t, mod, func(t *testing.T, env *Env) {
		env.OpenFile("a/a.go")
		env.OpenFile("b/b.go")
		env.AfterChange(
			// The Go command sometimes tells us about only one of the import cycle
			// errors below. For robustness of this test, succeed if we get either.
			//
			// TODO(golang/go#52904): we should get *both* of these errors.
			AnyOf(
				Diagnostics(env.AtRegexp("a/a.go", `"mod.test/b"`), WithMessage("import cycle")),
				Diagnostics(env.AtRegexp("b/b.go", `"mod.test/a"`), WithMessage("import cycle")),
			),
		)
		env.RegexpReplace("b/b.go", `const B = a\.B`, "")
		env.SaveBuffer("b/b.go")
		env.AfterChange(
			NoDiagnostics(ForFile("a/a.go")),
			NoDiagnostics(ForFile("b/b.go")),
		)
	})
}

func TestBadImport(t *testing.T) {
	const mod = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

import (
	_ "nosuchpkg"
)
`
	t.Run("module", func(t *testing.T) {
		Run(t, mod, func(t *testing.T, env *Env) {
			env.OnceMet(
				InitialWorkspaceLoad,
				Diagnostics(env.AtRegexp("main.go", `"nosuchpkg"`), WithMessage(`could not import nosuchpkg (no required module provides package "nosuchpkg"`)),
			)
		})
	})
	t.Run("GOPATH", func(t *testing.T) {
		WithOptions(
			InGOPATH(),
			EnvVars{"GO111MODULE": "off"},
			Modes(Default),
		).Run(t, mod, func(t *testing.T, env *Env) {
			env.OnceMet(
				InitialWorkspaceLoad,
				Diagnostics(env.AtRegexp("main.go", `"nosuchpkg"`), WithMessage(`cannot find package "nosuchpkg"`)),
			)
		})
	})
}

func TestNestedModules(t *testing.T) {
	const proxy = `
-- nested.com@v1.0.0/go.mod --
module nested.com

go 1.12
-- nested.com@v1.0.0/hello/hello.go --
package hello

func Hello() {}
`

	const nested = `
-- go.mod --
module mod.com

go 1.12

require nested.com v1.0.0
-- go.sum --
nested.com v1.0.0 h1:I6spLE4CgFqMdBPc+wTV2asDO2QJ3tU0YAT+jkLeN1I=
nested.com v1.0.0/go.mod h1:ly53UzXQgVjSlV7wicdBB4p8BxfytuGT1Xcyv0ReJfI=
-- main.go --
package main

import "nested.com/hello"

func main() {
	hello.Hello()
}
-- nested/go.mod --
module nested.com

-- nested/hello/hello.go --
package hello

func Hello() {
	helloHelper()
}
-- nested/hello/hello_helper.go --
package hello

func helloHelper() {}
`
	WithOptions(
		ProxyFiles(proxy),
		Modes(Default),
	).Run(t, nested, func(t *testing.T, env *Env) {
		// Expect a diagnostic in a nested module.
		env.OpenFile("nested/hello/hello.go")
		env.AfterChange(
			Diagnostics(env.AtRegexp("nested/hello/hello.go", "helloHelper")),
			Diagnostics(env.AtRegexp("nested/hello/hello.go", "package (hello)"), WithMessage("not included in your workspace")),
		)
	})
}

func TestAdHocPackagesReloading(t *testing.T) {
	const nomod = `
-- main.go --
package main

func main() {}
`
	Run(t, nomod, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.RegexpReplace("main.go", "{}", "{ var x int; }") // simulate typing
		env.AfterChange(NoLogMatching(protocol.Info, "packages=1"))
	})
}

func TestBuildTagChange(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.12
-- foo.go --
// decoy comment
// +build hidden
// decoy comment

package foo
var Foo = 1
-- bar.go --
package foo
var Bar = Foo
`

	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("foo.go")
		env.AfterChange(Diagnostics(env.AtRegexp("bar.go", `Foo`)))
		env.RegexpReplace("foo.go", `\+build`, "")
		env.AfterChange(NoDiagnostics(ForFile("bar.go")))
	})

}

func TestIssue44736(t *testing.T) {
	const files = `
	-- go.mod --
module blah.com

go 1.16
-- main.go --
package main

import "fmt"

func main() {
	asdf
	fmt.Printf("This is a test %v")
	fdas
}
-- other.go --
package main

`
	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.OpenFile("other.go")
		env.AfterChange(
			Diagnostics(env.AtRegexp("main.go", "asdf")),
			Diagnostics(env.AtRegexp("main.go", "fdas")),
		)
		env.SetBufferContent("other.go", "package main\n\nasdf")
		// The new diagnostic in other.go should not suppress diagnostics in main.go.
		env.AfterChange(
			Diagnostics(env.AtRegexp("other.go", "asdf"), WithMessage("expected declaration")),
			Diagnostics(env.AtRegexp("main.go", "asdf")),
		)
	})
}

func TestInitialization(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.16
-- main.go --
package main
`
	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("go.mod")
		env.Await(env.DoneWithOpen())
		env.RegexpReplace("go.mod", "module", "modul")
		env.SaveBufferWithoutActions("go.mod")
		env.AfterChange(
			NoLogMatching(protocol.Error, "initial workspace load failed"),
		)
	})
}

// This test confirms that the view does not reinitialize when a go.mod file is
// opened.
func TestNoReinitialize(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

func main() {}
`
	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("go.mod")
		env.Await(
			// Check that we have only loaded "<dir>/..." once.
			// Log messages are asynchronous to other events on the LSP stream, so we
			// can't use OnceMet or AfterChange here.
			LogMatching(protocol.Info, `.*query=.*\.\.\..*`, 1, false),
		)
	})
}

func TestLangVersion(t *testing.T) {
	testenv.NeedsGo1Point(t, 18) // Requires types.Config.GoVersion, new in 1.18.
	const files = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

const C = 0b10
`
	Run(t, files, func(t *testing.T, env *Env) {
		env.OnceMet(
			InitialWorkspaceLoad,
			Diagnostics(env.AtRegexp("main.go", `0b10`), WithMessage("go1.13 or later")),
		)
		env.WriteWorkspaceFile("go.mod", "module mod.com \n\ngo 1.13\n")
		env.AfterChange(
			NoDiagnostics(ForFile("main.go")),
		)
	})
}

func TestNoQuickFixForUndeclaredConstraint(t *testing.T) {
	testenv.NeedsGo1Point(t, 18)
	const files = `
-- go.mod --
module mod.com

go 1.18
-- main.go --
package main

func F[T C](_ T) {
}
`

	Run(t, files, func(t *testing.T, env *Env) {
		var d protocol.PublishDiagnosticsParams
		env.OnceMet(
			InitialWorkspaceLoad,
			Diagnostics(env.AtRegexp("main.go", `C`)),
			ReadDiagnostics("main.go", &d),
		)
		if fixes := env.GetQuickFixes("main.go", d.Diagnostics); len(fixes) != 0 {
			t.Errorf("got quick fixes %v, wanted none", fixes)
		}
	})
}

func TestEditGoDirective(t *testing.T) {
	testenv.NeedsGo1Point(t, 18)
	const files = `
-- go.mod --
module mod.com

go 1.16
-- main.go --
package main

func F[T any](_ T) {
}
`
	Run(t, files, func(_ *testing.T, env *Env) { // Create a new workspace-level directory and empty file.
		var d protocol.PublishDiagnosticsParams
		env.OnceMet(
			InitialWorkspaceLoad,
			Diagnostics(env.AtRegexp("main.go", `T any`), WithMessage("type parameter")),
			ReadDiagnostics("main.go", &d),
		)

		env.ApplyQuickFixes("main.go", d.Diagnostics)
		env.AfterChange(
			NoDiagnostics(ForFile("main.go")),
		)
	})
}

func TestEditGoDirectiveWorkspace(t *testing.T) {
	testenv.NeedsGo1Point(t, 18)
	const files = `
-- go.mod --
module mod.com

go 1.16
-- go.work --
go 1.18

use .
-- main.go --
package main

func F[T any](_ T) {
}
`
	Run(t, files, func(_ *testing.T, env *Env) { // Create a new workspace-level directory and empty file.
		var d protocol.PublishDiagnosticsParams

		// We should have a diagnostic because generics are not supported at 1.16.
		env.OnceMet(
			InitialWorkspaceLoad,
			Diagnostics(env.AtRegexp("main.go", `T any`), WithMessage("type parameter")),
			ReadDiagnostics("main.go", &d),
		)

		// This diagnostic should have a quick fix to edit the go version.
		env.ApplyQuickFixes("main.go", d.Diagnostics)

		// Once the edit is applied, the problematic diagnostics should be
		// resolved.
		env.AfterChange(
			NoDiagnostics(ForFile("main.go")),
		)
	})
}

// This test demonstrates that analysis facts are correctly propagated
// across packages.
func TestInterpackageAnalysis(t *testing.T) {
	const src = `
-- go.mod --
module example.com
-- a/a.go --
package a

import "example.com/b"

func _() {
	new(b.B).Printf("%d", "s") // printf error
}

-- b/b.go --
package b

import "example.com/c"

type B struct{}

func (B) Printf(format string, args ...interface{}) {
	c.MyPrintf(format, args...)
}

-- c/c.go --
package c

import "fmt"

func MyPrintf(format string, args ...interface{}) {
	fmt.Printf(format, args...)
}
`
	Run(t, src, func(t *testing.T, env *Env) {
		env.OpenFile("a/a.go")
		env.AfterChange(
			Diagnostics(
				env.AtRegexp("a/a.go", "new.*Printf"),
				WithMessage("format %d has arg \"s\" of wrong type string"),
			),
		)
	})
}

// This test ensures that only Analyzers with RunDespiteErrors=true
// are invoked on a package that would not compile, even if the errors
// are distant and localized.
func TestErrorsThatPreventAnalysis(t *testing.T) {
	const src = `
-- go.mod --
module example.com
-- a/a.go --
package a

import "fmt"
import "sync"
import _ "example.com/b"

func _() {
	// The copylocks analyzer (RunDespiteErrors, FactTypes={}) does run.
	var mu sync.Mutex
	mu2 := mu // copylocks error, reported
	_ = &mu2

	// The printf analyzer (!RunDespiteErrors, FactTypes!={}) does not run:
	//  (c, printf) failed because of type error in c
	//  (b, printf) and (a, printf) do not run because of failed prerequisites.
	fmt.Printf("%d", "s") // printf error, unreported

	// The bools analyzer (!RunDespiteErrors, FactTypes={}) does not run:
	var cond bool
	_ = cond != true && cond != true // bools error, unreported
}

-- b/b.go --
package b

import _ "example.com/c"

-- c/c.go --
package c

var _ = 1 / "" // type error

`
	Run(t, src, func(t *testing.T, env *Env) {
		var diags protocol.PublishDiagnosticsParams
		env.OpenFile("a/a.go")
		env.AfterChange(
			Diagnostics(env.AtRegexp("a/a.go", "mu2 := (mu)"), WithMessage("assignment copies lock value")),
			ReadDiagnostics("a/a.go", &diags))

		// Assert that there were no other diagnostics.
		// In particular:
		// - "fmt.Printf" does not trigger a [printf] finding;
		// - "cond != true" does not trigger a [bools] finding.
		//
		// We use this check in preference to NoDiagnosticAtRegexp
		// as it is robust in case of minor mistakes in the position
		// regexp, and because it reports unexpected diagnostics.
		if got, want := len(diags.Diagnostics), 1; got != want {
			t.Errorf("got %d diagnostics in a/a.go, want %d:", got, want)
			for i, diag := range diags.Diagnostics {
				t.Logf("Diagnostics[%d] = %+v", i, diag)
			}
		}
	})
}

// This test demonstrates the deprecated symbol analyzer
// produces deprecation notices with expected severity and tags.
func TestDeprecatedAnalysis(t *testing.T) {
	const src = `
-- go.mod --
module example.com
-- a/a.go --
package a

import "example.com/b"

func _() {
	new(b.B).Obsolete() // deprecated
}

-- b/b.go --
package b

type B struct{}

// Deprecated: use New instead.
func (B) Obsolete() {}

func (B) New() {}
`
	Run(t, src, func(t *testing.T, env *Env) {
		env.OpenFile("a/a.go")
		env.AfterChange(
			Diagnostics(
				env.AtRegexp("a/a.go", "new.*Obsolete"),
				WithMessage("use New instead."),
				WithSeverityTags("deprecated", protocol.SeverityHint, []protocol.DiagnosticTag{protocol.Deprecated}),
			),
		)
	})
}
