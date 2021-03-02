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

const filesA = `
-- go.mod --
module mod.com

go 1.12
-- b.gotmpl --
{{define "A"}}goo{{end}}
-- a.tmpl --
{{template "A"}}
`

func TestSuffixes(t *testing.T) {
	WithOptions(
		EditorConfig{
			AllExperiments: true,
		},
	).Run(t, filesA, func(t *testing.T, env *Env) {
		env.OpenFile("a.tmpl")
		x := env.RegexpSearch("a.tmpl", `A`)
		file, pos := env.GoToDefinition("a.tmpl", x)
		refs := env.References(file, pos)
		if len(refs) != 2 {
			t.Fatalf("got %v reference(s), want 2", len(refs))
		}
		// make sure we got one from b.gotmpl
		want := env.Sandbox.Workdir.URI("b.gotmpl")
		if refs[0].URI != want && refs[1].URI != want {
			t.Errorf("failed to find reference to %s", shorten(want))
			for i, r := range refs {
				t.Logf("%d: URI:%s %v", i, shorten(r.URI), r.Range)
			}
		}

		content, npos := env.Hover(file, pos)
		if pos != npos {
			t.Errorf("pos? got %v, wanted %v", npos, pos)
		}
		if content.Value != "template A defined" {
			t.Errorf("got %s, wanted 'template A defined", content.Value)
		}
	})
}

// shorten long URIs
func shorten(fn protocol.DocumentURI) string {
	if len(fn) <= 20 {
		return string(fn)
	}
	pieces := strings.Split(string(fn), "/")
	if len(pieces) < 2 {
		return string(fn)
	}
	j := len(pieces)
	return pieces[j-2] + "/" + pieces[j-1]
}

// Hover,  SemTok, Diagnose with errors
// and better coverage
