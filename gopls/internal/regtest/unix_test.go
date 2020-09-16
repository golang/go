// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows,!plan9

package regtest

import (
	"testing"
)

func TestBadGOPATH(t *testing.T) {
	const files = `
-- main.go --
package main

func _() {
	fmt.Println("Hello World")
}
`
	// https://github.com/fatih/vim-go/issues/2673#issuecomment-622307211.
	withOptions(
		EditorConfig{Env: map[string]string{"GOPATH": ":/path/to/gopath"}},
	).run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.Await(env.DiagnosticAtRegexp("main.go", "fmt"))
		if err := env.Editor.OrganizeImports(env.Ctx, "main.go"); err != nil {
			t.Fatal(err)
		}
	})
}
