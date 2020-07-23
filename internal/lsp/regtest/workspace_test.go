// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"testing"
)

const workspaceProxy = `
-- example.com@v1.2.3/go.mod --
module example.com

go 1.12
-- example.com@v1.2.3/blah/blah.go --
package blah

func SaySomething() {
	fmt.Println("something")
}
`

// TODO: Add a replace directive.
const workspaceModule = `
-- go.mod --
module mod.com

go 1.14

require example.com v1.2.3
-- main.go --
package main

import (
	"example.com/blah"
	"mod.com/inner"
)

func main() {
	blah.SaySomething()
	inner.Hi()
}
-- main2.go --
package main

import "fmt"

func _() {
	fmt.Print("%s")
}
-- inner/inner.go --
package inner

import "example.com/blah"

func Hi() {
	blah.SaySomething()
}
`

// Confirm that find references returns all of the references in the module,
// regardless of what the workspace root is.
func TestReferences(t *testing.T) {
	for _, tt := range []struct {
		name, rootPath string
	}{
		{
			name: "module root",
		},
		{
			name:     "subdirectory",
			rootPath: "inner",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			opts := []RunOption{WithProxyFiles(workspaceProxy)}
			if tt.rootPath != "" {
				opts = append(opts, WithRootPath(tt.rootPath))
			}
			withOptions(opts...).run(t, workspaceModule, func(t *testing.T, env *Env) {
				env.OpenFile("inner/inner.go")
				locations := env.ReferencesAtRegexp("inner/inner.go", "SaySomething")
				want := 3
				if got := len(locations); got != want {
					t.Fatalf("expected %v locations, got %v", want, got)
				}
			})
		})
	}
}

// Make sure that analysis diagnostics are cleared for the whole package when
// the only opened file is closed. This test was inspired by the experience in
// VS Code, where clicking on a reference result triggers a
// textDocument/didOpen without a corresponding textDocument/didClose.
func TestClearAnalysisDiagnostics(t *testing.T) {
	withOptions(WithProxyFiles(workspaceProxy), WithRootPath("inner")).run(t, workspaceModule, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.Await(
			env.DiagnosticAtRegexp("main2.go", "fmt.Print"),
		)
		env.CloseBuffer("main.go")
		env.Await(
			EmptyDiagnostics("main2.go"),
		)
	})
}
