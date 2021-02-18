// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"testing"

	. "golang.org/x/tools/gopls/internal/regtest"

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

type Info struct {
	WordCounts map[string]int
	Words []string
}

func Foo() {
	_ = Info{}
}
`
	Run(t, basic, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		pos := env.RegexpSearch("main.go", "Info{}").ToProtocolPosition()
		if err := env.Editor.RefactorRewrite(env.Ctx, "main.go", &protocol.Range{
			Start: pos,
			End:   pos,
		}); err != nil {
			t.Fatal(err)
		}
		want := `package main

type Info struct {
	WordCounts map[string]int
	Words []string
}

func Foo() {
	_ = Info{
		WordCounts: map[string]int{},
		Words:      []string{},
	}
}
`
		if got := env.Editor.BufferText("main.go"); got != want {
			t.Fatalf("TestFillStruct failed:\n%s", tests.Diff(t, want, got))
		}
	})
}
