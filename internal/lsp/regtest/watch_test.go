// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"testing"

	"golang.org/x/tools/internal/lsp"
	"golang.org/x/tools/internal/lsp/protocol"
)

func TestEditFile(t *testing.T) {
	const pkg = `
-- go.mod --
module mod.com

go 1.14
-- a/a.go --
package a

func _() {
	var x int
}
`
	// Edit the file when it's *not open* in the workspace, and check that
	// diagnostics are updated.
	t.Run("unopened", func(t *testing.T) {
		runner.Run(t, pkg, func(t *testing.T, env *Env) {
			env.Await(
				env.DiagnosticAtRegexp("a/a.go", "x"),
			)
			env.WriteWorkspaceFile("a/a.go", `package a; func _() {};`)
			env.Await(
				EmptyDiagnostics("a/a.go"),
			)
		})
	})

	// Edit the file when it *is open* in the workspace, and check that
	// diagnostics are *not* updated.
	t.Run("opened", func(t *testing.T) {
		runner.Run(t, pkg, func(t *testing.T, env *Env) {
			env.OpenFile("a/a.go")
			env.Await(CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidOpen), 1))
			env.WriteWorkspaceFile("a/a.go", `package a; func _() {};`)
			env.Await(
				env.DiagnosticAtRegexp("a/a.go", "x"),
			)
		})
	})
}

// Edit a dependency on disk and expect a new diagnostic.
func TestEditDependency(t *testing.T) {
	const pkg = `
-- go.mod --
module mod.com

go 1.14
-- b/b.go --
package b

func B() int { return 0 }
-- a/a.go --
package a

import (
	"mod.com/b"
)

func _() {
	_ = b.B()
}
`
	runner.Run(t, pkg, func(t *testing.T, env *Env) {
		env.OpenFile("a/a.go")
		env.Await(CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidOpen), 1))
		env.WriteWorkspaceFile("b/b.go", `package b; func B() {};`)
		env.Await(
			env.DiagnosticAtRegexp("a/a.go", "b.B"),
		)
	})
}

// Edit both the current file and one of its dependencies on disk and
// expect diagnostic changes.
func TestEditFileAndDependency(t *testing.T) {
	const pkg = `
-- go.mod --
module mod.com

go 1.14
-- b/b.go --
package b

func B() int { return 0 }
-- a/a.go --
package a

import (
	"mod.com/b"
)

func _() {
	var x int
	_ = b.B()
}
`
	runner.Run(t, pkg, func(t *testing.T, env *Env) {
		env.Await(
			env.DiagnosticAtRegexp("a/a.go", "x"),
		)
		env.WriteWorkspaceFiles(map[string]string{
			"b/b.go": `package b; func B() {};`,
			"a/a.go": `package a

import "mod.com/b"

func _() {
	b.B()
}`})
		env.Await(
			EmptyDiagnostics("a/a.go"),
			NoDiagnostics("b/b.go"),
		)
	})
}

// Delete a dependency and expect a new diagnostic.
func TestDeleteDependency(t *testing.T) {
	const pkg = `
-- go.mod --
module mod.com

go 1.14
-- b/b.go --
package b

func B() int { return 0 }
-- a/a.go --
package a

import (
	"mod.com/b"
)

func _() {
	_ = b.B()
}
`
	runner.Run(t, pkg, func(t *testing.T, env *Env) {
		env.OpenFile("a/a.go")
		env.Await(CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidOpen), 1))
		env.RemoveWorkspaceFile("b/b.go")
		env.Await(
			env.DiagnosticAtRegexp("a/a.go", "\"mod.com/b\""),
		)
	})
}

// Create a dependency on disk and expect the diagnostic to go away.
func TestCreateDependency(t *testing.T) {
	const missing = `
-- go.mod --
module mod.com

go 1.14
-- b/b.go --
package b

func B() int { return 0 }
-- a/a.go --
package a

import (
	"mod.com/c"
)

func _() {
	c.C()
}
`
	runner.Run(t, missing, func(t *testing.T, env *Env) {
		t.Skipf("the initial workspace load fails and never retries")

		env.Await(
			env.DiagnosticAtRegexp("a/a.go", "\"mod.com/c\""),
		)
		env.WriteWorkspaceFile("c/c.go", `package c; func C() {};`)
		env.Await(
			EmptyDiagnostics("c/c.go"),
		)
	})
}

// Create a new dependency and add it to the file on disk.
// This is similar to what might happen if you switch branches.
func TestCreateAndAddDependency(t *testing.T) {
	const original = `
-- go.mod --
module mod.com

go 1.14
-- a/a.go --
package a

func _() {}
`
	runner.Run(t, original, func(t *testing.T, env *Env) {
		env.WriteWorkspaceFile("c/c.go", `package c; func C() {};`)
		env.WriteWorkspaceFile("a/a.go", `package a; import "mod.com/c"; func _() { c.C() }`)
		env.Await(
			NoDiagnostics("a/a.go"),
		)
	})

}

// Create a new file that defines a new symbol, in the same package.
func TestCreateFile(t *testing.T) {
	const pkg = `
-- go.mod --
module mod.com

go 1.14
-- a/a.go --
package a

func _() {
	hello()
}
`
	runner.Run(t, pkg, func(t *testing.T, env *Env) {
		env.Await(
			env.DiagnosticAtRegexp("a/a.go", "hello"),
		)
		env.WriteWorkspaceFile("a/a2.go", `package a; func hello() {};`)
		env.Await(
			EmptyDiagnostics("a/a.go"),
		)
	})
}

// Add a new method to an interface and implement it.
// Inspired by the structure of internal/lsp/source and internal/lsp/cache.
func TestCreateImplementation(t *testing.T) {
	const pkg = `
-- go.mod --
module mod.com

go 1.14
-- b/b.go --
package b

type B interface{
	Hello() string
}

func SayHello(bee B) {
	println(bee.Hello())
}
-- a/a.go --
package a

import "mod.com/b"

type X struct {}

func (_ X) Hello() string {
	return ""
}

func _() {
	x := X{}
	b.SayHello(x)
}
`
	const newMethod = `package b
type B interface{
	Hello() string
	Bye() string
}

func SayHello(bee B) {
	println(bee.Hello())
}`
	const implementation = `package a

import "mod.com/b"

type X struct {}

func (_ X) Hello() string {
	return ""
}

func (_ X) Bye() string {
	return ""
}

func _() {
	x := X{}
	b.SayHello(x)
}`

	// Add the new method before the implementation. Expect diagnostics.
	t.Run("method before implementation", func(t *testing.T) {
		runner.Run(t, pkg, func(t *testing.T, env *Env) {
			env.Await(
				CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromInitialWorkspaceLoad), 1),
			)
			env.WriteWorkspaceFile("b/b.go", newMethod)
			env.Await(
				OnceMet(
					CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidChangeWatchedFiles), 1),
					DiagnosticAt("a/a.go", 12, 12),
				),
			)
			env.WriteWorkspaceFile("a/a.go", implementation)
			env.Await(
				EmptyDiagnostics("a/a.go"),
			)
		})
	})
	// Add the new implementation before the new method. Expect no diagnostics.
	t.Run("implementation before method", func(t *testing.T) {
		runner.Run(t, pkg, func(t *testing.T, env *Env) {
			env.Await(
				CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromInitialWorkspaceLoad), 1),
			)
			env.WriteWorkspaceFile("a/a.go", implementation)
			env.Await(
				OnceMet(
					CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidChangeWatchedFiles), 1),
					NoDiagnostics("a/a.go"),
				),
			)
			env.WriteWorkspaceFile("b/b.go", newMethod)
			env.Await(
				NoDiagnostics("a/a.go"),
			)
		})
	})
	// Add both simultaneously. Expect no diagnostics.
	t.Run("implementation and method simultaneously", func(t *testing.T) {
		runner.Run(t, pkg, func(t *testing.T, env *Env) {
			env.Await(
				CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromInitialWorkspaceLoad), 1),
			)
			env.WriteWorkspaceFiles(map[string]string{
				"a/a.go": implementation,
				"b/b.go": newMethod,
			})
			env.Await(
				OnceMet(
					CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidChangeWatchedFiles), 1),
					NoDiagnostics("a/a.go"),
				),
				NoDiagnostics("b/b.go"),
			)
		})
	})
}

// Tests golang/go#38498. Delete a file and then force a reload.
// Assert that we no longer try to load the file.
func TestDeleteFiles(t *testing.T) {
	const pkg = `
-- go.mod --
module mod.com

go 1.14
-- a/a.go --
package a

func _() {
	var _ int
}
-- a/a_unneeded.go --
package a
`
	t.Run("close then delete", func(t *testing.T) {
		runner.Run(t, pkg, func(t *testing.T, env *Env) {
			env.OpenFile("a/a.go")
			env.OpenFile("a/a_unneeded.go")
			env.Await(CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidOpen), 2))

			// Close and delete the open file, mimicking what an editor would do.
			env.CloseBuffer("a/a_unneeded.go")
			env.RemoveWorkspaceFile("a/a_unneeded.go")
			env.RegexpReplace("a/a.go", "var _ int", "fmt.Println(\"\")")
			env.Await(
				env.DiagnosticAtRegexp("a/a.go", "fmt"),
			)
			env.SaveBuffer("a/a.go")
			env.Await(
				NoLogMatching(protocol.Info, "a_unneeded.go"),
				EmptyDiagnostics("a/a.go"),
			)
		})
	})

	t.Run("delete then close", func(t *testing.T) {
		runner.Run(t, pkg, func(t *testing.T, env *Env) {
			env.OpenFile("a/a.go")
			env.OpenFile("a/a_unneeded.go")
			env.Await(CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidOpen), 2))

			// Delete and then close the file.
			env.CloseBuffer("a/a_unneeded.go")
			env.RemoveWorkspaceFile("a/a_unneeded.go")
			env.RegexpReplace("a/a.go", "var _ int", "fmt.Println(\"\")")
			env.Await(
				env.DiagnosticAtRegexp("a/a.go", "fmt"),
			)
			env.SaveBuffer("a/a.go")
			env.Await(
				NoLogMatching(protocol.Info, "a_unneeded.go"),
				EmptyDiagnostics("a/a.go"),
			)
		})
	})

}
