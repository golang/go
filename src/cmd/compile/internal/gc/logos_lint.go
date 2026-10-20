// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/noder"
)

// logosLintAfterEscape runs the lints that read escape analysis, then ends a
// -logos.lintonly compile: escape analysis is the last phase a lint rule
// reads, and the compile stops before SSA and the backend and writes no
// object file. Findings queued so far print here, and any error among them
// fails the compile.
func logosLintAfterEscape() {
	noder.LogosLintEscape()
	if !base.LogosLintOnly {
		return
	}
	base.ExitIfErrors()
	base.FlushErrors()
	base.Exit(0)
}
