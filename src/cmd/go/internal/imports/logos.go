// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package imports

import (
	"path/filepath"
	"strings"

	"cmd/go/internal/cfg"
)

// isLogosTestFile reports whether name is a loGo/s test source file,
// such as p_test.lgo or a p_test file with an alias extension.
// Logos-gated so plain go never classifies .lgo tests. Selectors sit
// after the extension, so the stem check needs no stripping of its own;
// cfg.IsLogosFile handles recognition.
func isLogosTestFile(name string) bool {
	if !cfg.Logos || !cfg.IsLogosFile(name) {
		return false
	}
	stem, _ := strings.CutSuffix(name, filepath.Ext(name))
	return strings.HasSuffix(stem, "_test")
}

// isTestFile reports whether name is a Go or loGo/s test source file.
func isTestFile(name string) bool {
	if strings.HasSuffix(name, "_test.go") {
		return true
	}
	return isLogosTestFile(name)
}
