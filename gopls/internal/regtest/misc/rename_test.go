// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"strings"
	"testing"

	"golang.org/x/tools/internal/lsp/protocol"
	. "golang.org/x/tools/internal/lsp/regtest"
)

func TestPrepareRenamePackage(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.18
-- main.go --
package main

import (
	"fmt"
)

func main() {
	fmt.Println(1)
}
`
	const wantErr = "can't rename packages: LSP client does not support file renaming"
	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		pos := env.RegexpSearch("main.go", `main`)
		tdpp := protocol.TextDocumentPositionParams{
			TextDocument: env.Editor.TextDocumentIdentifier("main.go"),
			Position:     pos.ToProtocolPosition(),
		}
		params := &protocol.PrepareRenameParams{
			TextDocumentPositionParams: tdpp,
		}
		_, err := env.Editor.Server.PrepareRename(env.Ctx, params)
		if err == nil {
			t.Errorf("missing can't rename package error from PrepareRename")
		}

		if err.Error() != wantErr {
			t.Errorf("got %v, want %v", err.Error(), wantErr)
		}
	})
}

// Test for golang/go#47564.
func TestRenameInTestVariant(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.12
-- stringutil/stringutil.go --
package stringutil

func Identity(s string) string {
	return s
}
-- stringutil/stringutil_test.go --
package stringutil

func TestIdentity(t *testing.T) {
	if got := Identity("foo"); got != "foo" {
		t.Errorf("bad")
	}
}
-- main.go --
package main

import (
	"fmt"

	"mod.com/stringutil"
)

func main() {
	fmt.Println(stringutil.Identity("hello world"))
}
`

	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		pos := env.RegexpSearch("main.go", `stringutil\.(Identity)`)
		env.Rename("main.go", pos, "Identityx")
		text := env.Editor.BufferText("stringutil/stringutil_test.go")
		if !strings.Contains(text, "Identityx") {
			t.Errorf("stringutil/stringutil_test.go: missing expected token `Identityx` after rename:\n%s", text)
		}
	})
}
