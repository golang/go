// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"fmt"
	"testing"

	. "golang.org/x/tools/gopls/internal/lsp/regtest"
)

func TestSubdirWatchPatterns(t *testing.T) {
	const files = `
-- go.mod --
module mod.test

go 1.18
-- subdir/subdir.go --
package subdir
`

	tests := []struct {
		clientName          string
		subdirWatchPatterns string
		wantWatched         bool
	}{
		{"other client", "on", true},
		{"other client", "off", false},
		{"other client", "auto", false},
		{"Visual Studio Code", "auto", true},
	}

	for _, test := range tests {
		t.Run(fmt.Sprintf("%s_%s", test.clientName, test.subdirWatchPatterns), func(t *testing.T) {
			WithOptions(
				ClientName(test.clientName),
				Settings{
					"subdirWatchPatterns": test.subdirWatchPatterns,
				},
			).Run(t, files, func(t *testing.T, env *Env) {
				var expectation Expectation
				if test.wantWatched {
					expectation = FileWatchMatching("subdir")
				} else {
					expectation = NoFileWatchMatching("subdir")
				}
				env.OnceMet(
					InitialWorkspaceLoad,
					expectation,
				)
			})
		})
	}
}

// This test checks that we surface errors for invalid subdir watch patterns,
// as the triple of ("off"|"on"|"auto") may be confusing to users inclined to
// use (true|false) or some other truthy value.
func TestSubdirWatchPatterns_BadValues(t *testing.T) {
	tests := []struct {
		badValue    interface{}
		wantMessage string
	}{
		{true, "invalid type bool, expect string"},
		{false, "invalid type bool, expect string"},
		{"yes", `invalid option "yes"`},
	}

	for _, test := range tests {
		t.Run(fmt.Sprint(test.badValue), func(t *testing.T) {
			WithOptions(
				Settings{
					"subdirWatchPatterns": test.badValue,
				},
			).Run(t, "", func(t *testing.T, env *Env) {
				env.OnceMet(
					InitialWorkspaceLoad,
					ShownMessage(test.wantMessage),
				)
			})
		})
	}
}
