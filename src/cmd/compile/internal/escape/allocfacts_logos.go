// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
)

var LogosFuncAllocs map[LogosFuncKey]bool

func logosRecordFuncAlloc(fn *ir.Func) {
	if LogosFuncAllocs == nil {
		return
	}
	for fn.ClosureParent != nil {
		fn = fn.ClosureParent
	}
	pos := base.Ctxt.InnermostPos(fn.Pos())
	LogosFuncAllocs[LogosFuncKey{File: pos.RelFilename(), Line: pos.Line()}] = true
}
