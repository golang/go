// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"testing"
)

func TestLICM(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("sp", OpSP, c.config.Types.Uintptr, 0, nil),
			Valu("a", OpConst64, c.config.Types.Int64, 14, nil),
			Goto("loop")),
		Bloc("loop",
			Valu("b", OpConst64, c.config.Types.Int64, 26, nil),
			Valu("sum", OpAdd64, c.config.Types.Int64, 0, nil, "a", "b"),
			Valu("load", OpLoad, c.config.Types.BytePtr, 0, nil, "sp", "mem"),
			Valu("nilptr", OpConstNil, c.config.Types.BytePtr, 0, nil),
			Valu("bool", OpNeqPtr, c.config.Types.Bool, 0, nil, "load", "nilptr"),
			If("bool", "loop", "exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)
	licm(fun.f)
	CheckFunc(fun.f)

	b := fun.blocks["entry"]
	if len(b.Values) != 5 {
		// b,sum should have been moved from loop to entry
		t.Errorf("loop invariant code wasn't lifted, but should have")
	}
}

func TestLICMNewBlock(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("sp", OpSP, c.config.Types.Uintptr, 0, nil),
			Valu("a", OpConst64, c.config.Types.Int64, 14, nil),
			Valu("bool2", OpConstBool, c.config.Types.Bool, 0, nil),
			If("bool2", "loop", "exit")),
		Bloc("loop",
			Valu("b", OpConst64, c.config.Types.Int64, 26, nil),
			Valu("sum", OpAdd64, c.config.Types.Int64, 0, nil, "a", "b"),
			Valu("load", OpLoad, c.config.Types.BytePtr, 0, nil, "sp", "mem"),
			Valu("nilptr", OpConstNil, c.config.Types.BytePtr, 0, nil),
			Valu("bool", OpNeqPtr, c.config.Types.Bool, 0, nil, "load", "nilptr"),
			If("bool", "loop", "exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)
	licm(fun.f)
	CheckFunc(fun.f)

	b := fun.blocks["entry"].Succs[0].b
	if len(b.Values) != 2 {
		// b,sum should have been moved from loop to new block between entry & loop
		t.Errorf("loop invariant code wasn't lifted, but should have")
	}
}
