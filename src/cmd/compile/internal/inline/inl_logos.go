// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inline

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
)

var logosRecursiveMarks = make(map[*ir.Func]bool)

func logosNoteSCC(funcs []*ir.Func) {
	for _, fn := range funcs {
		logosRecursiveMarks[fn] = true
	}
}

func logosBudgetFor(fn *ir.Func, def int32) int32 {
	if base.Debug.InlBudget == 0 {
		return def
	}
	if base.Flag.CompilingRuntime {
		return def
	}
	if logosRecursiveMarks[fn] {
		if base.Debug.InlRecursiveBudget != 0 {
			return int32(base.Debug.InlRecursiveBudget)
		}
		return def
	}
	return int32(base.Debug.InlBudget)
}

func logosConservativeBudget(fn *ir.Func) int32 {
	if base.Debug.InlBudget == 0 || base.Flag.CompilingRuntime || fn == nil {
		return 0
	}
	return int32(inlineMaxBudget)
}

func logosConservativeCallCost(calleeCost, extraCost int32) int32 {
	if calleeCost > extraCost {
		return extraCost
	}
	return calleeCost
}
