// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	. "golang.org/x/tools/gopls/internal/lsp/regtest"
)

func TestSignatureHelpInNonWorkspacePackage(t *testing.T) {
	const files = `
-- a/go.mod --
module a.com

go 1.18
-- a/a/a.go --
package a

func DoSomething(int) {}

func _() {
	DoSomething()
}
-- b/go.mod --
module b.com
go 1.18

require a.com v1.0.0

replace a.com => ../a
-- b/b/b.go --
package b

import "a.com/a"

func _() {
	a.DoSomething()
}
`

	WithOptions(
		WorkspaceFolders("a"),
	).Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("a/a/a.go")
		env.OpenFile("b/b/b.go")
		signatureHelp := func(filename string) *protocol.SignatureHelp {
			loc := env.RegexpSearch(filename, `DoSomething\(()\)`)
			var params protocol.SignatureHelpParams
			params.TextDocument.URI = loc.URI
			params.Position = loc.Range.Start
			help, err := env.Editor.Server.SignatureHelp(env.Ctx, &params)
			if err != nil {
				t.Fatal(err)
			}
			return help
		}
		ahelp := signatureHelp("a/a/a.go")
		bhelp := signatureHelp("b/b/b.go")

		if diff := cmp.Diff(ahelp, bhelp); diff != "" {
			t.Fatal(diff)
		}
	})
}
