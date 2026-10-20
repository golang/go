// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/noder"
)

// logosLintAfterEscape runs the lints that read escape analysis, then ends a
// -logos.lintonly or -logos.mode=hints compile: escape analysis is the last
// phase a lint rule or an escape hint reads, and the compile stops before SSA
// and the backend and writes no object file. Hints mode writes its report
// first. Findings queued so far print here, and any error among them fails
// the compile.
func logosLintAfterEscape() {
	noder.LogosLintEscape()
	if base.LogosMode == "hints" {
		noder.LogosHintsWrite()
		logosStopAfterEscape()
	}
	if base.LogosLintOnly {
		logosStopAfterEscape()
	}
}

// logosStopAfterEscape prints the queued findings and ends the compile with
// status zero unless one of them is an error.
func logosStopAfterEscape() {
	base.ExitIfErrors()
	base.FlushErrors()
	base.Exit(0)
}
