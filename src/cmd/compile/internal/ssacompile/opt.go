// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssarewrite/rewritedivisible"
	"cmd/compile/internal/ssarewrite/rewritedivmod"
	"cmd/compile/internal/ssarewrite/rewritegeneric"
)

// machine-independent optimization.
func opt(f *ssa.Func) {
	applyRewrite(f, rewritegeneric.RewriteBlock, rewritegeneric.RewriteValue, ssa.RemoveDeadValues)
}

func divisiblePass(f *ssa.Func) {
	applyRewrite(f, rewritedivisible.RewriteBlock, rewritedivisible.RewriteValue, ssa.RemoveDeadValues)
}

func divmodPass(f *ssa.Func) {
	applyRewrite(f, rewritedivmod.RewriteBlock, rewritedivmod.RewriteValue, ssa.RemoveDeadValues)
}
