// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2_test

import (
	"cmd/compile/internal/syntax"
	"strings"
)

// logosPackageMode reports the parse mode and Config.Logos value for a
// test package: a package of only .lgo files checks as loGo/s
// (syntax.LogosMode, true); anything else checks as Go (0, false), so
// every existing caller keeps its behavior. It lives here, not in
// check_test.go, because upstream files take hook lines only (I5);
// check_test.go keeps the TestLogos entry and the two call lines, with
// Config.Logos applied after the custom options (M1b).
func logosPackageMode(filenames []string) (syntax.Mode, bool) {
	for _, name := range filenames {
		if !strings.HasSuffix(name, ".lgo") {
			return 0, false
		}
	}
	return syntax.LogosMode, true
}
