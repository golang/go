// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package workspace

import (
	"testing"

	. "golang.org/x/tools/gopls/internal/lsp/regtest"
	"golang.org/x/tools/internal/testenv"
)

// Test for golang/go#57209: editing a file in an ad-hoc package should not
// trigger conflicting diagnostics.
func TestAdhoc_Edits(t *testing.T) {
	testenv.NeedsGo1Point(t, 18)

	const files = `
-- a.go --
package foo

const X = 1

-- b.go --
package foo

// import "errors"

const Y = X
`

	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("b.go")

		for i := 0; i < 10; i++ {
			env.RegexpReplace("b.go", `// import "errors"`, `import "errors"`)
			env.RegexpReplace("b.go", `import "errors"`, `// import "errors"`)
			env.AfterChange(NoDiagnostics())
		}
	})
}
