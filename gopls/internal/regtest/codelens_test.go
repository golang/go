// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"runtime"
	"strings"
	"testing"

	"golang.org/x/tools/internal/lsp"
	"golang.org/x/tools/internal/lsp/fake"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/tests"
	"golang.org/x/tools/internal/testenv"
)

func TestDisablingCodeLens(t *testing.T) {
	const workspace = `
-- go.mod --
module codelens.test

go 1.12
-- lib.go --
package lib

type Number int

const (
	Zero Number = iota
	One
	Two
)

//go:generate stringer -type=Number
`
	tests := []struct {
		label        string
		enabled      map[string]bool
		wantCodeLens bool
	}{
		{
			label:        "default",
			wantCodeLens: true,
		},
		{
			label:        "generate disabled",
			enabled:      map[string]bool{source.CommandGenerate.Name: false},
			wantCodeLens: false,
		},
	}
	for _, test := range tests {
		t.Run(test.label, func(t *testing.T) {
			withOptions(
				EditorConfig{
					CodeLenses: test.enabled,
				},
			).run(t, workspace, func(t *testing.T, env *Env) {
				env.OpenFile("lib.go")
				lens := env.CodeLens("lib.go")
				if gotCodeLens := len(lens) > 0; gotCodeLens != test.wantCodeLens {
					t.Errorf("got codeLens: %t, want %t", gotCodeLens, test.wantCodeLens)
				}
			})
		})
	}
}

// This test confirms the full functionality of the code lenses for updating
// dependencies in a go.mod file. It checks for the code lens that suggests
// an update and then executes the command associated with that code lens. A
// regression test for golang/go#39446.
func TestUpdateCodelens(t *testing.T) {
	const proxyWithLatest = `
-- golang.org/x/hello@v1.3.3/go.mod --
module golang.org/x/hello

go 1.12
-- golang.org/x/hello@v1.3.3/hi/hi.go --
package hi

var Goodbye error
	-- golang.org/x/hello@v1.2.3/go.mod --
module golang.org/x/hello

go 1.12
-- golang.org/x/hello@v1.2.3/hi/hi.go --
package hi

var Goodbye error
`

	const shouldUpdateDep = `
-- go.mod --
module mod.com

go 1.12

require golang.org/x/hello v1.2.3
-- go.sum --
golang.org/x/hello v1.2.3 h1:jOtNXLsiCuLzU6KM3wRHidpc29IxcKpofHZiOW1hYKA=
golang.org/x/hello v1.2.3/go.mod h1:X79D30QqR94cGK8aIhQNhCZLq4mIr5Gimj5qekF08rY=
-- main.go --
package main

import "golang.org/x/hello/hi"

func main() {
	_ = hi.Goodbye
}
`
	runner.Run(t, shouldUpdateDep, func(t *testing.T, env *Env) {
		env.OpenFile("go.mod")
		env.ExecuteCodeLensCommand("go.mod", source.CommandUpgradeDependency)
		env.Await(CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidChangeWatchedFiles), 1))
		got := env.Editor.BufferText("go.mod")
		const wantGoMod = `module mod.com

go 1.12

require golang.org/x/hello v1.3.3
`
		if got != wantGoMod {
			t.Fatalf("go.mod upgrade failed:\n%s", tests.Diff(wantGoMod, got))
		}
	}, WithProxyFiles(proxyWithLatest))
}

func TestUnusedDependenciesCodelens(t *testing.T) {
	testenv.NeedsGo1Point(t, 14)
	const proxy = `
-- golang.org/x/hello@v1.0.0/go.mod --
module golang.org/x/hello

go 1.14
-- golang.org/x/hello@v1.0.0/hi/hi.go --
package hi

var Goodbye error
-- golang.org/x/unused@v1.0.0/go.mod --
module golang.org/x/unused

go 1.14
-- golang.org/x/unused@v1.0.0/nouse/nouse.go --
package nouse

var NotUsed error
`

	const shouldRemoveDep = `
-- go.mod --
module mod.com

go 1.14

require golang.org/x/hello v1.0.0
require golang.org/x/unused v1.0.0
-- go.sum --
golang.org/x/hello v1.0.0 h1:qbzE1/qT0/zojAMd/JcPsO2Vb9K4Bkeyq0vB2JGMmsw=
golang.org/x/hello v1.0.0/go.mod h1:WW7ER2MRNXWA6c8/4bDIek4Hc/+DofTrMaQQitGXcco=
golang.org/x/unused v1.0.0 h1:LecSbCn5P3vTcxubungSt1Pn4D/WocCaiWOPDC0y0rw=
golang.org/x/unused v1.0.0/go.mod h1:ihoW8SgWzugwwj0N2SfLfPZCxTB1QOVfhMfB5PWTQ8U=
-- main.go --
package main

import "golang.org/x/hello/hi"

func main() {
	_ = hi.Goodbye
}
`
	runner.Run(t, shouldRemoveDep, func(t *testing.T, env *Env) {
		env.OpenFile("go.mod")
		env.ExecuteCodeLensCommand("go.mod", source.CommandTidy)
		env.Await(CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidChangeWatchedFiles), 1))
		got := env.Editor.BufferText("go.mod")
		const wantGoMod = `module mod.com

go 1.14

require golang.org/x/hello v1.0.0
`
		if got != wantGoMod {
			t.Fatalf("go.mod tidy failed:\n%s", tests.Diff(wantGoMod, got))
		}
	}, WithProxyFiles(proxy))
}

func TestRegenerateCgo(t *testing.T) {
	testenv.NeedsTool(t, "cgo")
	testenv.NeedsGo1Point(t, 15)

	const workspace = `
-- go.mod --
module example.com

go 1.12
-- cgo.go --
package x

/*
int fortythree() { return 42; }
*/
import "C"

func Foo() {
	print(C.fortytwo())
}
`
	runner.Run(t, workspace, func(t *testing.T, env *Env) {
		// Open the file. We should have a nonexistant symbol.
		env.OpenFile("cgo.go")
		env.Await(env.DiagnosticAtRegexp("cgo.go", `C\.(fortytwo)`)) // could not determine kind of name for C.fortytwo

		// Fix the C function name. We haven't regenerated cgo, so nothing should be fixed.
		env.RegexpReplace("cgo.go", `int fortythree`, "int fortytwo")
		env.SaveBuffer("cgo.go")
		env.Await(env.DiagnosticAtRegexp("cgo.go", `C\.(fortytwo)`))

		// Regenerate cgo, fixing the diagnostic.
		env.ExecuteCodeLensCommand("cgo.go", source.CommandRegenerateCgo)
		env.Await(EmptyDiagnostics("cgo.go"))
	})
}

func TestGCDetails(t *testing.T) {
	testenv.NeedsGo1Point(t, 15)
	if runtime.GOOS == "android" {
		t.Skipf("the gc details code lens doesn't work on Android")
	}

	const mod = `
-- go.mod --
module mod.com

go 1.15
-- main.go --
package main

import "fmt"

func main() {
	var x string
	fmt.Println(x)
}
`
	withOptions(
		EditorConfig{
			CodeLenses: map[string]bool{
				"gc_details": true,
			}},
	).run(t, mod, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.ExecuteCodeLensCommand("main.go", source.CommandToggleDetails)
		d := &protocol.PublishDiagnosticsParams{}
		env.Await(
			OnceMet(
				DiagnosticAt("main.go", 6, 12),
				ReadDiagnostics("main.go", d),
			),
		)
		// Confirm that the diagnostics come from the gc details code lens.
		var found bool
		for _, d := range d.Diagnostics {
			if d.Severity != protocol.SeverityInformation {
				t.Fatalf("unexpected diagnostic severity %v, wanted Information", d.Severity)
			}
			if strings.Contains(d.Message, "x escapes") {
				found = true
			}
		}
		if !found {
			t.Fatalf(`expected to find diagnostic with message "escape(x escapes to heap)", found none`)
		}

		// Editing a buffer should cause gc_details diagnostics to disappear, since
		// they only apply to saved buffers.
		env.EditBuffer("main.go", fake.NewEdit(0, 0, 0, 0, "\n\n"))
		env.Await(EmptyDiagnostics("main.go"))

		// Saving a buffer should re-format back to the original state, and
		// re-enable the gc_details diagnostics.
		env.SaveBuffer("main.go")
		env.Await(DiagnosticAt("main.go", 6, 12))

		// Toggle the GC details code lens again so now it should be off.
		env.ExecuteCodeLensCommand("main.go", source.CommandToggleDetails)
		env.Await(
			EmptyDiagnostics("main.go"),
		)
	})
}
