// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/noder"
)

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

func logosStopAfterEscape() {
	base.ExitIfErrors()
	base.FlushErrors()
	base.Exit(0)
}
