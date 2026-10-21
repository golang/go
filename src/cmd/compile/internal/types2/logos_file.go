// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"path/filepath"
	"strings"

	"cmd/compile/internal/syntax"
)

// isLogosFile reports whether the code at at follows the loGo/s rules:
// the package is a loGo/s package (Config.Logos) and at does not lie in a
// cgo-generated Go file (M20-S0). Every loGo/s rule gates on it, so the
// per-file decision is keyed on the file name, never on the package.
func (check *Checker) isLogosFile(at poser) bool {
	return check.conf.Logos && !logosGoModePos(at.Pos())
}

// logosGoModePos reports whether pos lies in a cgo-generated Go file of a
// loGo/s package: "_cgo_gotypes.go" and "_cgo_import.go". cmd/cgo writes
// them in Go and the compiler checks them with the package's .lgo files,
// so their declarations keep Go semantics. go/build ignores user files
// that start with "_", so the prefix names generated files only. The
// rewritten user file of a cgo input ("c_bridge.lgo.cgo1.go") holds the
// package's own code and follows the loGo/s rules. The file base, not a
// //line base, names the compiler input.
func logosGoModePos(pos syntax.Pos) bool {
	name := filepath.Base(pos.FileBase().Filename())
	return strings.HasPrefix(name, "_cgo_") && strings.HasSuffix(name, ".go")
}
