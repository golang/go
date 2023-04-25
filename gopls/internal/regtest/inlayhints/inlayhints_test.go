// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package inlayhint

import (
	"testing"

	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/hooks"
	. "golang.org/x/tools/gopls/internal/lsp/regtest"
	"golang.org/x/tools/gopls/internal/lsp/source"
)

func TestMain(m *testing.M) {
	bug.PanicOnBugs = true
	Main(m, hooks.Options)
}

func TestEnablingInlayHints(t *testing.T) {
	const workspace = `
-- go.mod --
module inlayHint.test
go 1.12
-- lib.go --
package lib
type Number int
const (
	Zero Number = iota
	One
	Two
)
`
	tests := []struct {
		label         string
		enabled       map[string]bool
		wantInlayHint bool
	}{
		{
			label:         "default",
			wantInlayHint: false,
		},
		{
			label:         "enable const",
			enabled:       map[string]bool{source.ConstantValues: true},
			wantInlayHint: true,
		},
		{
			label:         "enable parameter names",
			enabled:       map[string]bool{source.ParameterNames: true},
			wantInlayHint: false,
		},
	}
	for _, test := range tests {
		t.Run(test.label, func(t *testing.T) {
			WithOptions(
				Settings{
					"hints": test.enabled,
				},
			).Run(t, workspace, func(t *testing.T, env *Env) {
				env.OpenFile("lib.go")
				lens := env.InlayHints("lib.go")
				if gotInlayHint := len(lens) > 0; gotInlayHint != test.wantInlayHint {
					t.Errorf("got inlayHint: %t, want %t", gotInlayHint, test.wantInlayHint)
				}
			})
		})
	}
}
