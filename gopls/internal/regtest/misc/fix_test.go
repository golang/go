// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"testing"

	. "golang.org/x/tools/internal/lsp/regtest"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/tests"
)

// A basic test for fillstruct, now that it uses a command.
func TestFillStruct(t *testing.T) {
	const basic = `
-- go.mod --
module mod.com

go 1.14
-- main.go --
package main

type Info struct {
	WordCounts map[string]int
	Words []string
}

func Foo() {
	_ = Info{}
}
`
	Run(t, basic, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		pos := env.RegexpSearch("main.go", "Info{}").ToProtocolPosition()
		if err := env.Editor.RefactorRewrite(env.Ctx, "main.go", &protocol.Range{
			Start: pos,
			End:   pos,
		}); err != nil {
			t.Fatal(err)
		}
		want := `package main

type Info struct {
	WordCounts map[string]int
	Words []string
}

func Foo() {
	_ = Info{
		WordCounts: map[string]int{},
		Words:      []string{},
	}
}
`
		if got := env.Editor.BufferText("main.go"); got != want {
			t.Fatalf("TestFillStruct failed:\n%s", tests.Diff(t, want, got))
		}
	})
}

func TestFillReturns(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

func Foo() error {
	return
}
`
	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		var d protocol.PublishDiagnosticsParams
		env.Await(OnceMet(
			env.DiagnosticAtRegexpWithMessage("main.go", `return`, "wrong number of return values"),
			ReadDiagnostics("main.go", &d),
		))
		codeActions := env.CodeAction("main.go", d.Diagnostics)
		if len(codeActions) != 2 {
			t.Fatalf("expected 2 code actions, got %v", len(codeActions))
		}
		var foundQuickFix, foundFixAll bool
		for _, a := range codeActions {
			if a.Kind == protocol.QuickFix {
				foundQuickFix = true
			}
			if a.Kind == protocol.SourceFixAll {
				foundFixAll = true
			}
		}
		if !foundQuickFix {
			t.Fatalf("expected quickfix code action, got none")
		}
		if !foundFixAll {
			t.Fatalf("expected fixall code action, got none")
		}
		env.ApplyQuickFixes("main.go", d.Diagnostics)
		env.Await(EmptyDiagnostics("main.go"))
	})
}
