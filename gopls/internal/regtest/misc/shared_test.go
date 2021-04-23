// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"testing"

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

func runShared(t *testing.T, testFunc func(env1 *Env, env2 *Env)) {
	// Only run these tests in forwarded modes.
	modes := DefaultModes() & (Forwarded | SeparateProcess)
	WithOptions(Modes(modes)).Run(t, sharedProgram, func(t *testing.T, env1 *Env) {
		// Create a second test session connected to the same workspace and server
		// as the first.
		env2 := NewEnv(env1.Ctx, t, env1.Sandbox, env1.Server, env1.Editor.Config, true)
		testFunc(env1, env2)
	})
}

func TestSimultaneousEdits(t *testing.T) {
	runShared(t, func(env1 *Env, env2 *Env) {
		// In editor #1, break fmt.Println as before.
		env1.OpenFile("main.go")
		env1.RegexpReplace("main.go", "Printl(n)", "")
		// In editor #2 remove the closing brace.
		env2.OpenFile("main.go")
		env2.RegexpReplace("main.go", "\\)\n(})", "")

		// Now check that we got different diagnostics in each environment.
		env1.Await(env1.DiagnosticAtRegexp("main.go", "Printl"))
		env2.Await(env2.DiagnosticAtRegexp("main.go", "$"))
	})
}

func TestShutdown(t *testing.T) {
	runShared(t, func(env1 *Env, env2 *Env) {
		env1.CloseEditor()
		// Now make an edit in editor #2 to trigger diagnostics.
		env2.OpenFile("main.go")
		env2.RegexpReplace("main.go", "\\)\n(})", "")
		env2.Await(env2.DiagnosticAtRegexp("main.go", "$"))
	})
}
