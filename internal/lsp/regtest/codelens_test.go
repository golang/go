// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"testing"

	"golang.org/x/tools/internal/lsp/fake"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
)

func TestDisablingCodeLens(t *testing.T) {
	const workspace = `
-- go.mod --
module codelens.test
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
			enabled:      map[string]bool{source.CommandGenerate: false},
			wantCodeLens: false,
		},
	}
	for _, test := range tests {
		t.Run(test.label, func(t *testing.T) {
			runner.Run(t, workspace, func(t *testing.T, env *Env) {
				env.OpenFile("lib.go")
				lens := env.CodeLens("lib.go")
				if gotCodeLens := len(lens) > 0; gotCodeLens != test.wantCodeLens {
					t.Errorf("got codeLens: %t, want %t", gotCodeLens, test.wantCodeLens)
				}
			}, WithEditorConfig(fake.EditorConfig{CodeLens: test.enabled}))
		})
	}
}

// This test confirms the full functionality of the code lenses for updating
// dependencies in a go.mod file. It checks for the code lens that suggests an
// update and then executes the command associated with that code lens.
// A regression test for golang/go#39446.
func TestUpdateCodelens(t *testing.T) {
	const proxyWithLatest = `
-- golang.org/x/hello@v1.3.3/go.mod --
module golang.org/x/hello

go 1.14
-- golang.org/x/hello@v1.3.3/hi/hi.go --
package hi

var Goodbye error
	-- golang.org/x/hello@v1.2.3/go.mod --
module golang.org/x/hello

go 1.14
-- golang.org/x/hello@v1.2.3/hi/hi.go --
package hi

var Goodbye error
`

	const shouldUpdateDep = `
-- go.mod --
module mod.com

go 1.14

require golang.org/x/hello v1.2.3
-- main.go --
package main

import "golang.org/x/hello/hi"

func main() {
	_ = hi.Goodbye
}
`
	runner.Run(t, shouldUpdateDep, func(t *testing.T, env *Env) {
		env.OpenFile("go.mod")
		lenses := env.CodeLens("go.mod")
		want := "Upgrade dependency to v1.3.3"
		var found *protocol.CodeLens
		for _, lens := range lenses {
			if lens.Command.Title == want {
				found = &lens
			}
		}
		if found == nil {
			t.Fatalf("did not find lens %q, got %v", want, lenses)
		}
		if _, err := env.Editor.Server.ExecuteCommand(env.Ctx, &protocol.ExecuteCommandParams{
			Command:   found.Command.Command,
			Arguments: found.Command.Arguments,
		}); err != nil {
			t.Fatal(err)
		}
	}, WithProxy(proxyWithLatest))
}
