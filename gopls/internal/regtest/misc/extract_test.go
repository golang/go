// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package misc

import (
	"testing"

	. "golang.org/x/tools/gopls/internal/lsp/regtest"
	"golang.org/x/tools/gopls/internal/lsp/tests/compare"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
)

func TestExtractFunction(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

func Foo() int {
	a := 5
	return a
}
`
	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		loc := env.RegexpSearch("main.go", `a := 5\n.*return a`)
		actions, err := env.Editor.CodeAction(env.Ctx, loc, nil)
		if err != nil {
			t.Fatal(err)
		}

		// Find the extract function code action.
		var extractFunc *protocol.CodeAction
		for _, action := range actions {
			if action.Kind == protocol.RefactorExtract && action.Title == "Extract function" {
				extractFunc = &action
				break
			}
		}
		if extractFunc == nil {
			t.Fatal("could not find extract function action")
		}

		env.ApplyCodeAction(*extractFunc)
		want := `package main

func Foo() int {
	return newFunction()
}

func newFunction() int {
	a := 5
	return a
}
`
		if got := env.BufferText("main.go"); got != want {
			t.Fatalf("TestFillStruct failed:\n%s", compare.Text(want, got))
		}
	})
}
