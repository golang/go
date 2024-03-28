// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"testing"
)

func doLCSSA(fun fun) bool {
	CheckFunc(fun.f)
	f := fun.f
	loopnest := f.loopnest()
	loopnest.assembleChildren()
	loopnest.findExits()
	for _, loop := range loopnest.loops {
		if f.BuildLoopClosedForm(loopnest, loop) {
			CheckFunc(fun.f)
			return true
		}
	}
	return false
}

// Simple Case: use block is dominated by a single loop exit
func TestLoopUse1(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("loopEntry",
		Bloc("loopEntry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("zero", OpConst64, c.config.Types.Int64, 0, nil),
			Valu("one", OpConst64, c.config.Types.Int64, 1, nil),
			Valu("ten", OpConst64, c.config.Types.Int64, 10, nil),
			Goto("loopHeader")),
		Bloc("loopHeader",
			Valu("i", OpPhi, c.config.Types.Int64, 0, nil, "zero", "inc"),
			Valu("cmp", OpLess64, c.config.Types.Bool, 0, nil, "i", "ten"),
			If("cmp", "loopLatch", "loopExit")),
		Bloc("loopLatch",
			Valu("inc", OpAdd64, c.config.Types.Int64, 0, nil, "one", "i"),
			Goto("loopHeader")),
		Bloc("loopExit",
			Goto("useBlock")),
		Bloc("useBlock",
			Valu("use", OpAdd64, c.config.Types.Int64, 0, nil, "one", "i"),
			Exit("mem")))

	if !doLCSSA(fun) {
		t.Fatal("Failed to build LCSSA")
	}

	//  loop header:
	//  i = phi(0, inc)
	//  ....
	//
	//  loop exit:
	//  p1 = phi(i) <= proxy phi
	//  Plain useBlock
	//
	//  useBlock:
	//  use = p1 + 1
	verifyNumValue(fun, t, OpPhi, 2 /*var i + 1 proxy phi*/)
}

// Harder Case: use block is reachable from multiple loop exits
func TestLoopUse2(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("loopEntry",
		Bloc("loopEntry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("zero", OpConst64, c.config.Types.Int64, 0, nil),
			Valu("one", OpConst64, c.config.Types.Int64, 1, nil),
			Valu("ten", OpConst64, c.config.Types.Int64, 10, nil),
			Goto("loopHeader")),
		Bloc("loopHeader",
			Valu("i", OpPhi, c.config.Types.Int64, 0, nil, "zero", "inc"),
			Valu("cmp", OpLess64, c.config.Types.Bool, 0, nil, "i", "ten"),
			If("cmp", "loopBody", "loopExit")),
		Bloc("loopBody",
			Valu("cmp2", OpEq64, c.config.Types.Bool, 0, nil, "i", "one"),
			If("cmp2", "loopExit1", "loopLatch")),
		Bloc("loopLatch",
			Valu("inc", OpAdd64, c.config.Types.Int64, 0, nil, "one", "i"),
			Goto("loopHeader")),
		Bloc("loopExit1",
			Goto("useBlock")),
		Bloc("loopExit",
			Goto("useBlock")),
		Bloc("useBlock",
			Valu("use", OpMul64, c.config.Types.Int64, 0, nil, "i", "ten"),
			Exit("mem")))

	if !doLCSSA(fun) {
		t.Fatal("Failed to build LCSSA")
	}

	//  loop header:
	//  i = phi(0, inc)
	//  ....
	//
	//  loop exit:
	//  p1 = phi(i) <= proxy phi
	//  Plain useBlock
	//
	//  loop exit1:
	//  p2 = phi(i) <= proxy phi
	//  Plain useBlock
	//
	//  useBlock:
	//  p3 = phi(p1, p2) <= proxy phi
	//  use = p1 + 1
	verifyNumValue(fun, t, OpPhi, 4 /*var i + 3 proxy phi*/)
}

// Used by ctrl valule
func TestLoopUse3(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("loopEntry",
		Bloc("loopEntry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("zero", OpConst64, c.config.Types.Int64, 0, nil),
			Valu("one", OpConst64, c.config.Types.Int64, 1, nil),
			Valu("ten", OpConst64, c.config.Types.Int64, 10, nil),
			Goto("loopHeader")),
		Bloc("loopHeader",
			Valu("i", OpPhi, c.config.Types.Int64, 0, nil, "zero", "inc"),
			Valu("cmp", OpLess64, c.config.Types.Bool, 0, nil, "i", "ten"),
			If("cmp", "loopLatch", "loopExit")),
		Bloc("loopLatch",
			Valu("inc", OpAdd64, c.config.Types.Int64, 0, nil, "one", "i"),
			Goto("loopHeader")),
		Bloc("loopExit",
			// used by ctrl value
			If("cmp", "exit1", "exit2")),
		Bloc("exit1",
			Goto("exit2")),
		Bloc("exit2",
			Exit("mem")))

	if !doLCSSA(fun) {
		t.Fatal("Failed to build LCSSA")
	}

	//  loop header:
	//  i = phi(0, inc)
	//  ....
	//
	//  loop exit:
	//  p1 = phi(i) <= proxy phi
	//  If p1-> exit1, exit2
	verifyNumValue(fun, t, OpPhi, 2 /*var i + 1 proxy phi*/)
}

// Used by Phi
func TestLoopUse4(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("loopEntry",
		Bloc("loopEntry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("zero", OpConst64, c.config.Types.Int64, 0, nil),
			Valu("one", OpConst64, c.config.Types.Int64, 1, nil),
			Valu("ten", OpConst64, c.config.Types.Int64, 10, nil),
			Goto("loopHeader")),
		Bloc("loopHeader",
			Valu("i", OpPhi, c.config.Types.Int64, 0, nil, "zero", "inc"),
			Valu("cmp", OpLess64, c.config.Types.Bool, 0, nil, "i", "ten"),
			If("cmp", "loopBody", "loopExit")),
		Bloc("loopBody",
			Valu("cmp2", OpEq64, c.config.Types.Bool, 0, nil, "i", "one"),
			If("cmp2", "loopExit1", "loopLatch")),
		Bloc("loopLatch",
			Valu("inc", OpAdd64, c.config.Types.Int64, 0, nil, "one", "i"),
			Goto("loopHeader")),
		Bloc("loopExit1",
			Goto("useBlock")),
		Bloc("loopExit",
			Goto("useBlock")),
		Bloc("useBlock",
			Valu("use", OpPhi, c.config.Types.Int64, 0, nil, "i", "i"),
			Exit("mem")))

	if !doLCSSA(fun) {
		t.Fatal("Failed to build LCSSA")
	}

	//  loop header:
	//  i = phi(0, inc)
	//  ....
	//
	//  loop exit:
	//  p1 = phi(i) <= proxy phi
	//  Plain useBlock
	//
	//  loop exit1:
	//  p2 = phi(i) <= proxy phi
	//  Plain useBlock
	//
	//  useBlock:
	//  use = phi(p1, p2)
	verifyNumValue(fun, t, OpPhi, 3 /*var i + 2 proxy phi*/ +1 /*original phi*/)
}
