// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows

package regtest

import (
	"fmt"
	"testing"

	"golang.org/x/tools/internal/lsp/fake"
)

func TestBadGOPATH(t *testing.T) {
	const missingImport = `
-- main.go --
package main

func _() {
	fmt.Println("Hello World")
}
`
	// Test the case given in
	// https://github.com/fatih/vim-go/issues/2673#issuecomment-622307211.
	runner.Run(t, missingImport, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.Await(env.DiagnosticAtRegexp("main.go", "fmt"))
		if err := env.Editor.OrganizeImports(env.Ctx, "main.go"); err != nil {
			t.Fatal(err)
		}
	}, WithEditorConfig(fake.EditorConfig{
		Env: []string{fmt.Sprintf("GOPATH=:/path/to/gopath")},
	}))
}
