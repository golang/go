// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"testing"

	. "golang.org/x/tools/internal/lsp/regtest"
)

func TestEmptyDirectoryFilters_Issue51843(t *testing.T) {
	const src = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

func main() {
}
`

	WithOptions(
		EditorConfig{
			Settings: map[string]interface{}{
				"directoryFilters": []string{""},
			},
		},
	).Run(t, src, func(t *testing.T, env *Env) {
		// No need to do anything. Issue golang/go#51843 is triggered by the empty
		// directory filter above.
	})
}
