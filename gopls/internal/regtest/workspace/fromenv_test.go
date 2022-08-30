// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package workspace

import (
	"testing"

	. "golang.org/x/tools/gopls/internal/lsp/regtest"
)

// Test that setting go.work via environment variables or settings works.
func TestUseGoWorkOutsideTheWorkspace(t *testing.T) {
	const files = `
-- work/a/go.mod --
module a.com

go 1.12
-- work/a/a.go --
package a
-- work/b/go.mod --
module b.com

go 1.12
-- work/b/b.go --
package b

func _() {
	x := 1 // unused
}
-- config/go.work --
go 1.18

use (
	$SANDBOX_WORKDIR/work/a
	$SANDBOX_WORKDIR/work/b
)
`

	WithOptions(
		EnvVars{"GOWORK": "$SANDBOX_WORKDIR/config/go.work"},
	).Run(t, files, func(t *testing.T, env *Env) {
		// When we have an explicit GOWORK set, we should get a file watch request.
		env.Await(FileWatchMatching(`config.go\.work`))
		// Even though work/b is not open, we should get its diagnostics as it is
		// included in the workspace.
		env.OpenFile("work/a/a.go")
		env.Await(
			OnceMet(
				env.DoneWithOpen(),
				env.DiagnosticAtRegexpWithMessage("work/b/b.go", "x := 1", "not used"),
			),
		)
	})
}
