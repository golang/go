// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/fake"
	. "golang.org/x/tools/gopls/internal/lsp/regtest"
)

// Smoke test that simultaneous editing sessions in the same workspace works.
func TestSimultaneousEdits(t *testing.T) {
	const sharedProgram = `
-- go.mod --
module mod

go 1.12
-- main.go --
package main

import "fmt"

func main() {
	fmt.Println("Hello World.")
}`

	WithOptions(
		Modes(DefaultModes()&(Forwarded|SeparateProcess)),
	).Run(t, sharedProgram, func(t *testing.T, env1 *Env) {
		// Create a second test session connected to the same workspace and server
		// as the first.
		awaiter := NewAwaiter(env1.Sandbox.Workdir)
		editor, err := fake.NewEditor(env1.Sandbox, env1.Editor.Config()).Connect(env1.Ctx, env1.Server, awaiter.Hooks())
		if err != nil {
			t.Fatal(err)
		}
		env2 := &Env{
			T:       t,
			Ctx:     env1.Ctx,
			Sandbox: env1.Sandbox,
			Server:  env1.Server,
			Editor:  editor,
			Awaiter: awaiter,
		}
		env2.Await(InitialWorkspaceLoad)
		// In editor #1, break fmt.Println as before.
		env1.OpenFile("main.go")
		env1.RegexpReplace("main.go", "Printl(n)", "")
		// In editor #2 remove the closing brace.
		env2.OpenFile("main.go")
		env2.RegexpReplace("main.go", "\\)\n(})", "")

		// Now check that we got different diagnostics in each environment.
		env1.Await(env1.DiagnosticAtRegexp("main.go", "Printl"))
		env2.Await(env2.DiagnosticAtRegexp("main.go", "$"))

		// Now close editor #2, and verify that operation in editor #1 is
		// unaffected.
		if err := env2.Editor.Close(env2.Ctx); err != nil {
			t.Errorf("closing second editor: %v", err)
		}

		env1.RegexpReplace("main.go", "Printl", "Println")
		env1.Await(
			OnceMet(
				env1.DoneWithChange(),
				EmptyDiagnostics("main.go"),
			),
		)
	})
}
