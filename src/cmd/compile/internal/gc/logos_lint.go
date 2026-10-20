// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import "cmd/compile/internal/base"

// logosLintStop ends a -logos.lintonly compile after escape analysis, the
// last phase a lint rule reads, and before SSA and the backend: the compile
// writes no object file. Findings queued so far print here, and any error
// among them fails the compile.
func logosLintStop() {
	if !base.LogosLintOnly {
		return
	}
	base.ExitIfErrors()
	base.FlushErrors()
	base.Exit(0)
}
