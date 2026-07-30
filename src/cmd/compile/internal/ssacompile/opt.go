// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import "cmd/compile/internal/ssa/ssacore"

// machine-independent optimization.
func opt(f *ssacore.Func) {
	applyRewrite(f, rewriteBlockgeneric, rewriteValuegeneric, RemoveDeadValues)
}

func divisible(f *ssacore.Func) {
	applyRewrite(f, rewriteBlockdivisible, rewriteValuedivisible, RemoveDeadValues)
}

func divmod(f *ssacore.Func) {
	applyRewrite(f, rewriteBlockdivmod, rewriteValuedivmod, RemoveDeadValues)
}
