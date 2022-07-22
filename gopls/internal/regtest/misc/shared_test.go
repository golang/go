// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"testing"

	"golang.org/x/tools/internal/lsp/fake"
	. "golang.org/x/tools/internal/lsp/regtest"
)

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

// runShared is a helper to run a test in the same directory using both the
// original env, and an additional other environment connected to the same
// server.
func runShared(t *testing.T, testFunc func(origEnv *Env, otherEnv *Env)) {
	// Only run these tests in forwarded modes.
	modes := DefaultModes() & (Forwarded | SeparateProcess)
	WithOptions(Modes(modes)).Run(t, sharedProgram, func(t *testing.T, env1 *Env) {
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
		testFunc(env1, env2)
	})
}

func TestSimultaneousEdits(t *testing.T) {
	runShared(t, func(origEnv *Env, otherEnv *Env) {
		// In editor #1, break fmt.Println as before.
		origEnv.OpenFile("main.go")
		origEnv.RegexpReplace("main.go", "Printl(n)", "")
		// In editor #2 remove the closing brace.
		otherEnv.OpenFile("main.go")
		otherEnv.RegexpReplace("main.go", "\\)\n(})", "")

		// Now check that we got different diagnostics in each environment.
		origEnv.Await(origEnv.DiagnosticAtRegexp("main.go", "Printl"))
		otherEnv.Await(otherEnv.DiagnosticAtRegexp("main.go", "$"))
	})
}

func TestShutdown(t *testing.T) {
	runShared(t, func(origEnv *Env, otherEnv *Env) {
		// Close otherEnv, and verify that operation in the original environment is
		// unaffected. Note: 'otherEnv' must be the environment being closed here.
		// If we were to instead close 'env' here, we'd run into a duplicate
		// shutdown when the test runner closes the original env.
		if err := otherEnv.Editor.Close(otherEnv.Ctx); err != nil {
			t.Errorf("closing first editor: %v", err)
		}
		// Now make an edit in editor #2 to trigger diagnostics.
		origEnv.OpenFile("main.go")
		origEnv.RegexpReplace("main.go", "\\)\n(})", "")
		origEnv.Await(origEnv.DiagnosticAtRegexp("main.go", "$"))
	})
}
