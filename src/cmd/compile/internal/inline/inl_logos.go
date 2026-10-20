// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inline

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
)

// logosRecursiveMarks records the functions of every recursive call-graph
// cycle that any CanInlineFuncs pass of this compile has seen, so the -O
// tier recursive budget and the recursive-cycle stack-buffer threshold
// (LogosVariableMakeThreshold) apply to them. The map only grows, and a
// pass never clears it: noder/reader.go runs CanInlineFuncs again on
// freshly instantiated generic functions after every inlined body that
// instantiates one, in the middle of inlining the package, so a per-pass
// reset dropped the marks of the package's own functions before the
// backend read them. Passes run on the single frontend goroutine and the
// map is read-only once the backend starts, so concurrent function
// compilation never races on it.
var logosRecursiveMarks = make(map[*ir.Func]bool)

// logosNoteSCC records one recursive call-graph cycle.
func logosNoteSCC(funcs []*ir.Func) {
	for _, fn := range funcs {
		logosRecursiveMarks[fn] = true
	}
}

// logosBudgetFor returns the inline budget for fn: the -O tier budget,
// the recursive-cycle budget inside recursive cycles, or the upstream
// default when no tier flag is set. The runtime package always keeps the
// upstream default: its nowritebarrier-checked code is validated at that
// budget, and a higher tier budget inlines write barriers into callers
// that forbid them.
func logosBudgetFor(fn *ir.Func, def int32) int32 {
	if base.Debug.InlBudget == 0 {
		return def
	}
	if base.Flag.CompilingRuntime {
		return def
	}
	if logosRecursiveMarks[fn] && base.Debug.InlRecursiveBudget != 0 {
		return int32(base.Debug.InlRecursiveBudget)
	}
	return int32(base.Debug.InlBudget)
}
