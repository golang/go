// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"testing"

	"golang.org/x/tools/internal/lsp/protocol"
	. "golang.org/x/tools/internal/lsp/regtest"
)

func TestBadURICrash_VSCodeIssue1498(t *testing.T) {
	const src = `
-- go.mod --
module example.com

go 1.12

-- main.go --
package main

func main() {}

`
	WithOptions(
		Modes(Singleton),
		EditorConfig{
			AllExperiments: true,
		},
	).Run(t, src, func(t *testing.T, env *Env) {
		params := &protocol.SemanticTokensParams{}
		const badURI = "http://foo"
		params.TextDocument.URI = badURI
		// This call panicked in the past: golang/vscode-go#1498.
		if _, err := env.Editor.Server.SemanticTokensFull(env.Ctx, params); err != nil {
			// Requests to an invalid URI scheme shouldn't result in an error, we
			// simply don't support this so return empty result. This could be
			// changed, but for now assert on the current behavior.
			t.Errorf("SemanticTokensFull(%q): %v", badURI, err)
		}
	})
}
