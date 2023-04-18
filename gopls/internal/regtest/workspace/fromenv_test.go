// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package workspace

import (
	"fmt"
	"path/filepath"
	"testing"

	. "golang.org/x/tools/gopls/internal/lsp/regtest"
	"golang.org/x/tools/internal/testenv"
)

// Test that setting go.work via environment variables or settings works.
func TestUseGoWorkOutsideTheWorkspace(t *testing.T) {
	testenv.NeedsGo1Point(t, 18)

	// As discussed in
	// https://github.com/golang/go/issues/59458#issuecomment-1513794691, we must
	// use \-separated paths in go.work use directives for this test to work
	// correctly on windows.
	var files = fmt.Sprintf(`
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
-- other/c/go.mod --
module c.com

go 1.18
-- other/c/c.go --
package c
-- config/go.work --
go 1.18

use (
	%s
	%s
	%s
)
`,
		filepath.Join("$SANDBOX_WORKDIR", "work", "a"),
		filepath.Join("$SANDBOX_WORKDIR", "work", "b"),
		filepath.Join("$SANDBOX_WORKDIR", "other", "c"),
	)

	WithOptions(
		WorkspaceFolders("work"), // use a nested workspace dir, so that GOWORK is outside the workspace
		EnvVars{"GOWORK": filepath.Join("$SANDBOX_WORKDIR", "config", "go.work")},
	).Run(t, files, func(t *testing.T, env *Env) {
		// When we have an explicit GOWORK set, we should get a file watch request.
		env.OnceMet(
			InitialWorkspaceLoad,
			FileWatchMatching(`other`),
			FileWatchMatching(`config.go\.work`),
		)
		env.Await(FileWatchMatching(`config.go\.work`))
		// Even though work/b is not open, we should get its diagnostics as it is
		// included in the workspace.
		env.OpenFile("work/a/a.go")
		env.AfterChange(
			Diagnostics(env.AtRegexp("work/b/b.go", "x := 1"), WithMessage("not used")),
		)
	})
}
