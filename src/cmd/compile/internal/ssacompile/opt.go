// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import "cmd/compile/internal/ssa"

// machine-independent optimization.
func opt(f *ssa.Func) {
	applyRewrite(f, rewriteBlockgeneric, rewriteValuegeneric, ssa.RemoveDeadValues)
}

func divisible(f *ssa.Func) {
	applyRewrite(f, rewriteBlockdivisible, rewriteValuedivisible, ssa.RemoveDeadValues)
}

func divmod(f *ssa.Func) {
	applyRewrite(f, rewriteBlockdivmod, rewriteValuedivmod, ssa.RemoveDeadValues)
}
