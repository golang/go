// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2_test

import (
	"cmd/compile/internal/syntax"
	"strings"
)

func logosPackageMode(filenames []string) (syntax.Mode, bool) {
	for _, name := range filenames {
		if !strings.HasSuffix(name, ".lgo") {
			return 0, false
		}
	}
	return syntax.LogosMode, true
}
