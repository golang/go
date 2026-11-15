// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inline

import (
	"testing"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

// TestLogosConservativeBudget verifies the policy governing conservative
// call-cost retry under different -O tiers.
func TestLogosConservativeBudget(t *testing.T) {
	oldInlBudget := base.Debug.InlBudget
	oldCompilingRuntime := base.Flag.CompilingRuntime
	t.Cleanup(func() {
		base.Debug.InlBudget = oldInlBudget
		base.Flag.CompilingRuntime = oldCompilingRuntime
	})

	base.Flag.CompilingRuntime = false
	fn := &ir.Func{}

	// O1 (base.Debug.InlBudget == 0): upstream behavior preserved, no retry.
	base.Debug.InlBudget = 0
	if got := logosConservativeBudget(fn); got != 0 {
		t.Errorf("O1: logosConservativeBudget(fn) = %d, want 0", got)
	}
	if got := logosConservativeBudget(nil); got != 0 {
		t.Errorf("O1: logosConservativeBudget(nil) = %d, want 0", got)
	}

	// O2 (base.Debug.InlBudget == 120): retry budget is upstream O1 budget (80).
	base.Debug.InlBudget = 120
	if got := logosConservativeBudget(fn); got != int32(inlineMaxBudget) {
		t.Errorf("O2: logosConservativeBudget(fn) = %d, want %d", got, inlineMaxBudget)
	}
	if got := logosConservativeBudget(nil); got != 0 {
		t.Errorf("O2: logosConservativeBudget(nil) = %d, want 0", got)
	}

	// O3 (base.Debug.InlBudget == 400): retry budget is upstream O1 budget (80).
	base.Debug.InlBudget = 400
	if got := logosConservativeBudget(fn); got != int32(inlineMaxBudget) {
		t.Errorf("O3: logosConservativeBudget(fn) = %d, want %d", got, inlineMaxBudget)
	}

	// CompilingRuntime: upstream behavior preserved, no retry.
	base.Flag.CompilingRuntime = true
	if got := logosConservativeBudget(fn); got != 0 {
		t.Errorf("CompilingRuntime: logosConservativeBudget(fn) = %d, want 0", got)
	}
}

// TestLogosConservativeCallCost verifies that logosConservativeCallCost
// caps expanded callee costs at extraCost when conservativeCalls is active,
// preventing wrapper inlining inversion while retaining cheaper callee costs.
func TestLogosConservativeCallCost(t *testing.T) {
	const extraCost = int32(57)

	for _, tc := range []struct {
		name       string
		calleeCost int32
		want       int32
	}{
		{"callee exceeds extraCost", 105, extraCost},
		{"callee equal to extraCost", extraCost, extraCost},
		{"callee below extraCost", 20, 20},
		{"callee zero cost", 0, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := logosConservativeCallCost(tc.calleeCost, extraCost); got != tc.want {
				t.Errorf("logosConservativeCallCost(%d, %d) = %d, want %d", tc.calleeCost, extraCost, got, tc.want)
			}
		})
	}
}

func TestLogosReflectLookupNoInline(t *testing.T) {
	oldHaveInlineBody := typecheck.HaveInlineBody
	typecheck.HaveInlineBody = func(*ir.Func) bool { return false }
	t.Cleanup(func() { typecheck.HaveInlineBody = oldHaveInlineBody })
	reflectPkg := types.NewPkg("reflect", "reflect")
	lookups := []string{
		"Value.Method",
		"Value.MethodByName",
		"(*rtype).Method",
		"(*rtype).MethodByName",
		"(*interfaceType).Method",
		"(*interfaceType).MethodByName",
	}
	for _, name := range lookups {
		fn := &ir.Func{Nname: ir.NewNameAt(src.NoXPos, reflectPkg.Lookup(name), nil)}
		reason := InlineImpossible(fn)
		if reason == "" || reason == "no function body" {
			t.Errorf("InlineImpossible(%q) = %q, want the reflect lookup block", name, reason)
		}
	}
	otherPkg := types.NewPkg("example.com/other", "other")
	other := &ir.Func{Nname: ir.NewNameAt(src.NoXPos, otherPkg.Lookup("Method"), nil)}
	if reason := InlineImpossible(other); reason != "no function body" {
		t.Errorf("InlineImpossible(other.Method) = %q, want %q", reason, "no function body")
	}
}
