// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"testing"

	"golang.org/x/tools/internal/lsp"
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

import "go/types"

func Foo() {
	_ = types.Info{}
}
`
	runner.Run(t, basic, func(t *testing.T, env *Env) {
		env.Await(
			CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromInitialWorkspaceLoad), 1),
		)
		env.OpenFile("main.go")
		if err := env.Editor.RefactorRewrite(env.Ctx, "main.go", &protocol.Range{
			Start: protocol.Position{
				Line:      5,
				Character: 16,
			},
			End: protocol.Position{
				Line:      5,
				Character: 16,
			},
		}); err != nil {
			t.Fatal(err)
		}
		want := `package main

import "go/types"

func Foo() {
	_ = types.Info{
		Types:      map[ast.Expr]types.TypeAndValue{},
		Defs:       map[*ast.Ident]types.Object{},
		Uses:       map[*ast.Ident]types.Object{},
		Implicits:  map[ast.Node]types.Object{},
		Selections: map[*ast.SelectorExpr]*types.Selection{},
		Scopes:     map[ast.Node]*types.Scope{},
		InitOrder:  []*types.Initializer{},
	}
}
`
		if got := env.Editor.BufferText("main.go"); got != want {
			t.Fatalf("TestFillStruct failed:\n%s", tests.Diff(want, got))
		}
	})
}
