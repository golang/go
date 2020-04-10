// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"testing"

	"golang.org/x/tools/internal/lsp/fake"
)

const symbolSetup = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

import "fmt"

func main() {
	fmt.Println(Message)
}
-- const.go --
package main

const Message = "Hello World."
`

// TestSymbolPos tests that, at a basic level, we get the correct position
// information for symbols matches that are returned.
func TestSymbolPos(t *testing.T) {
	matcher := "caseSensitive"
	opts := []RunOption{
		WithEditorConfig(fake.EditorConfig{SymbolMatcher: &matcher}),
	}

	runner.Run(t, symbolSetup, func(t *testing.T, env *Env) {
		res := env.Symbol("main")
		exp := &expSymbolInformation{
			Name: pString("main"),
			Location: &expLocation{
				Path: pString("main.go"),
				Range: &expRange{
					Start: &expPos{
						Line:   pInt(4),
						Column: pInt(5),
					},
				},
			},
		}
		if !exp.matchAgainst(res) {
			t.Fatalf("failed to find match for main function")
		}
	}, opts...)
}
