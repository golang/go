// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"internal/logosfile"
	"path/filepath"
	"strings"

	"cmd/compile/internal/syntax"
)

func (check *Checker) isLogosFile(at poser) bool {
	return check.conf.Logos && !logosGoModePos(at.Pos())
}

func logosGoModePos(pos syntax.Pos) bool {
	name := filepath.Base(pos.FileBase().Filename())
	return strings.HasPrefix(name, "_cgo_") && strings.HasSuffix(name, ".go")
}

func (check *Checker) logosFuncInLgo(fn *Func) bool {
	fn = fn.Origin()
	switch {
	case fn.pkg == nil:
		return false
	case fn.pkg == check.pkg:
		return !logosIsCgoName(fn.name) && !logosGoModePos(fn.pos)
	}
	base := fn.pos.FileBase()
	return base != nil && logosfile.Is(strings.TrimSuffix(base.Filename(), ".cgo1.go"))
}
