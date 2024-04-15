// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"testing"
)

func doLoopRotation(fun fun) bool {
	CheckFunc(fun.f)
	f := fun.f
	loopnest := f.loopnest()
	loopnest.assembleChildren()
	loopnest.findExits()
	for _, loop := range loopnest.loops {
		if !f.RotateLoop(loop) {
			return false
		}
		CheckFunc(fun.f)
	}
	return true
}

func doLoopRotationWithLCSSSA(fun fun) bool {
	CheckFunc(fun.f)
	f := fun.f
	loopnest := f.loopnest()
	loopnest.assembleChildren()
	loopnest.findExits()
	for _, loop := range loopnest.loops {
		if !f.BuildLoopClosedForm(loopnest, loop) {
			panic("Failed to build loop closed form")
		}
	}

	for _, loop := range loopnest.loops {
		if !f.RotateLoop(loop) {
			return false
		}
		CheckFunc(fun.f)
	}
	return true
}

func verifyRotatedCFG(fun fun, t *testing.T) {
	// CFG is correctly wired?
	cfg := map[string][]string{
		"loopHeader": {"loopLatch", "loopBody"},
		"loopLatch":  {"loopHeader", "loopExit"},
		"loopBody":   {"loopLatch"},
	}
	for k, succs := range cfg {
		for _, b := range fun.f.Blocks {
			if fun.blocks[k] == b {
				for _, succ := range succs {
					succb := fun.blocks[succ]
					found := false
					for _, s := range b.Succs {
						if s.b == succb {
							found = true
							break
						}
					}
					if !found {
						t.Fatalf("Illegal CFG")
					}
				}
			}
			break
		}
	}
}

func verifyNumValue(fun fun, t *testing.T, expectedOp Op, expectedNum int) {
	// Data flow is correctly set up?
	num := 0
	for _, b := range fun.f.Blocks {
		for _, val := range b.Values {
			if val.Op == expectedOp {
				num++
			}
		}
	}
	if num != expectedNum {
		t.Fatalf("unexpected num of operation %v", expectedOp)
	}
}

// The original loop looks like in below form
//
//	for i := 0; i < 10; i++ {
//	}
//
// After loop rotation, it should be like below
//
//	if 0 < 10 {
//		i := 0
//		do {
//			i++
//		} while i < 10
//	}
//
// Loop defs are not used outside the loop, so simply performing loop rotation
// w/o LCSSA is okay.
func TestSimpleLoop(t *testing.T) {
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
			Exit("mem")))

	if !doLoopRotation(fun) {
		t.Fatal("Loop rotation failed")
	}

	verifyRotatedCFG(fun, t)

	// one lives in loop latch and one lives in loop guard
	verifyNumValue(fun, t, OpLess64, 2)
}

// Loop header contains Values that may takes side effects and it was used by
// condiitonal test.
//
//	for i := 0; i < *load; i++ {
//	}
//
// After loop rotation, it should be like below
//
//	if 0 < *load {
//		i := 0
//		do {
//			i+=*load
//		} while *load < 10
//	}
//
// Loop defs are not used outside the loop, so simply performing loop rotation
// w/o LCSSA is okay.
func TestComplexLoop(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("loopEntry",
		Bloc("loopEntry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("zero", OpConst64, c.config.Types.Int64, 0, nil),
			Valu("one", OpConst64, c.config.Types.Int64, 1, nil),
			Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
			Goto("loopHeader")),
		Bloc("loopHeader",
			Valu("i", OpPhi, c.config.Types.Int64, 0, nil, "zero", "inc"),
			Valu("addr", OpAddr, c.config.Types.Int64.PtrTo(), 0, nil, "sb"),
			Valu("store", OpStore, types.TypeMem, 0, nil, "addr", "one", "mem"),
			Valu("load", OpLoad, c.config.Types.Int64, 0, nil, "addr", "store"),
			Valu("cmp", OpLess64, c.config.Types.Bool, 0, nil, "i", "load"),
			If("cmp", "loopLatch", "loopExit")),
		Bloc("loopLatch",
			Valu("inc", OpAdd64, c.config.Types.Int64, 0, nil, "load", "i"),
			Goto("loopHeader")),
		Bloc("loopExit",
			Exit("mem")))

	if !doLoopRotation(fun) {
		t.Fatal("Loop rotation failed")
	}

	verifyRotatedCFG(fun, t)

	// one lives in loop latch and one lives in loop guard
	verifyNumValue(fun, t, OpLess64, 2)
	verifyNumValue(fun, t, OpLoad, 2)
	verifyNumValue(fun, t, OpAddr, 2)
	verifyNumValue(fun, t, OpStore, 2)
}

// Similiar to TestSimpleLoop, but control value is not live in loop header
//
//	i := 0
//	cmp := i < 10
//	for ; cmp; i++ {
//	}
//
// After loop rotation, it should be like below
//
//	i := 0
//	cmp := i < 10
//	if cmp {
//		i := 0
//		do {
//			i++
//		} while cmp
//	}
//
// Loop defs are not used outside the loop, so simply performing loop rotation
// w/o LCSSA is okay.
func TestSimpleLoopCtrlElsewhere(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("loopEntry",
		Bloc("loopEntry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("zero", OpConst64, c.config.Types.Int64, 0, nil),
			Valu("one", OpConst64, c.config.Types.Int64, 1, nil),
			Valu("ten", OpConst64, c.config.Types.Int64, 10, nil),
			Valu("i", OpConst64, c.config.Types.Int64, 0, nil),
			Valu("cmp", OpLess64, c.config.Types.Bool, 0, nil, "i", "ten"),
			Goto("loopHeader")),
		Bloc("loopHeader",
			Valu("phi", OpPhi, c.config.Types.Int64, 0, nil, "i", "inc"),
			If("cmp", "loopLatch", "loopExit")),
		Bloc("loopLatch",
			Valu("inc", OpAdd64, c.config.Types.Int64, 0, nil, "one", "phi"),
			Goto("loopHeader")),
		Bloc("loopExit",
			Exit("mem")))

	if !doLoopRotation(fun) {
		t.Fatal("Loop rotation failed")
	}

	verifyRotatedCFG(fun, t)

	// no copy, no clone
	verifyNumValue(fun, t, OpLess64, 1)
	verifyNumValue(fun, t, OpPhi, 1)
}

// Even more harder, Values in loop header have cyclic dependencies, i.e.
//
// loop header:
//
//	v1 = phi(.., v3)
//	v3 = add(v1, 1)
//	If v3 < 10, ...
func TestCondCyclicLoop(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("loopEntry",
		Bloc("loopEntry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("zero", OpConst64, c.config.Types.Int64, 0, nil),
			Valu("one", OpConst64, c.config.Types.Int64, 1, nil),
			Valu("ten", OpConst64, c.config.Types.Int64, 10, nil),
			Goto("loopHeader")),
		Bloc("loopHeader",
			// cyclic dependency in loop header
			Valu("i", OpPhi, c.config.Types.Int64, 0, nil, "zero", "inc"),
			Valu("inc", OpAdd64, c.config.Types.Int64, 0, nil, "one", "i"),
			Valu("cmp", OpLess64, c.config.Types.Bool, 0, nil, "inc", "ten"),
			If("cmp", "loopLatch", "loopExit")),
		Bloc("loopLatch",
			Goto("loopHeader")),
		Bloc("loopExit",
			Exit("mem")))

	if !doLoopRotation(fun) {
		t.Fatal("Loop rotation failed")
	}

	verifyRotatedCFG(fun, t)

	// one lives in loop latch and one lives in loop guard
	verifyNumValue(fun, t, OpLess64, 2)
	verifyNumValue(fun, t, OpCopy, 1)
	verifyNumValue(fun, t, OpPhi, 2)

	for _, b := range fun.f.Blocks {
		for _, val := range b.Values {
			switch val.Op {
			case OpCopy:
				if val.Block != fun.blocks["loopLatch"] {
					t.Fatalf("copy must be in loop latch")
				}
			}
		}
	}
}

// Cyclic dependencies may appear during updating
//
//	loop header:
//	v1 = phi(.., v4)
//	v3 = add(v1, 1)
//	If v3 < 10, ...
//
//	loop latch:
//	v4 = add(v3, 1)
func TestNewCyclicLoop(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("loopEntry",
		Bloc("loopEntry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("zero", OpConst64, c.config.Types.Int64, 0, nil),
			Valu("one", OpConst64, c.config.Types.Int64, 1, nil),
			Valu("ten", OpConst64, c.config.Types.Int64, 10, nil),
			Goto("loopHeader")),
		Bloc("loopHeader",
			// cyclic dependency in loop header
			Valu("i", OpPhi, c.config.Types.Int64, 0, nil, "zero", "inc2"),
			Valu("inc", OpAdd64, c.config.Types.Int64, 0, nil, "one", "i"),
			Valu("cmp", OpLess64, c.config.Types.Bool, 0, nil, "inc", "ten"),
			If("cmp", "loopLatch", "loopExit")),
		Bloc("loopLatch",
			Valu("inc2", OpAdd64, c.config.Types.Int64, 0, nil, "one", "inc"),
			Goto("loopHeader")),
		Bloc("loopExit",
			Exit("mem")))

	if !doLoopRotation(fun) {
		t.Fatal("Loop rotation failed")
	}

	verifyRotatedCFG(fun, t)

	// one lives in loop latch and one lives in loop guard
	verifyNumValue(fun, t, OpLess64, 2)
	// no copy because inc2 explicitly uses inc
	verifyNumValue(fun, t, OpPhi, 2)
}

// Use loop phi outside the loop, this requires LCSSA, which creates proxy phi
// and use such phi outside the loop.
//
//	if 0 < 10 {
//		i := 0
//		do {
//			i++
//		} while i < 10
//		use := i * 10
//	}
func TestOutsideLoopUses(t *testing.T) {
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
			Valu("use", OpMul64, c.config.Types.Int64, 0, nil, "i", "ten"),
			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	// doLoopRotation fails because loop phi is used outside the loop.
	if !doLoopRotationWithLCSSSA(fun) {
		t.Fatal("Loop rotation failed")
	}

	verifyRotatedCFG(fun, t)

	// one lives in loop latch and one lives in loop guard
	verifyNumValue(fun, t, OpLess64, 2)

	loopExit := fun.blocks["loopExit"]
	for _, val := range loopExit.Values {
		if val.Op == OpPhi {
			if len(val.Args) != len(loopExit.Preds) {
				t.Fatalf("num of phi arguments mismatched with num of predecessors")
			}
			if 1 != val.Uses {
				t.Fatalf("proxy phi must be used by p")
			}
			for _, arg := range val.Args {
				switch arg.Op {
				case OpConst64, OpAdd64:
				default:
					t.Fatalf("proxy phi must have only constants and add operands")
				}
			}
		}
	}
}

// Ditto, but the loop phi has cyclic dependencies.
func TestPhiCondCyclicLoop(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("loopEntry",
		Bloc("loopEntry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("true", OpConstBool, c.config.Types.Bool, 1, nil),
			Goto("loopHeader")),
		Bloc("loopHeader",
			Valu("phi", OpPhi, c.config.Types.Bool, 0, nil, "true", "false"),
			Valu("false", OpConstBool, c.config.Types.Bool, 0, nil),
			If("phi", "loopLatch", "loopExit")),
		Bloc("loopLatch",
			Goto("loopHeader")),
		Bloc("loopExit",
			Exit("mem")))

	if !doLoopRotation(fun) {
		t.Fatal("Loop rotation failed")
	}

	verifyRotatedCFG(fun, t)
	// phi will not copy to loop guard, so only one phi exists
	verifyNumValue(fun, t, OpPhi, 1)
}

// Loop has multiple exits
//
//	for i := 0; i < 10; i++ {
//		if i == 1 {
//			return
//		}
//	}
//
// After loop rotation, it should be like below
//
//	if 0 < 10 {
//		i := 0
//		do {
//			if i == 1 {
//				return
//			}
//			i++
//		} while i < 10
//	}
//
// Loop defs are not used outside the loop, so simply performing loop rotation
// w/o LCSSA is okay.
func TestMultiExitLoop(t *testing.T) {
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
			Exit("mem")),
		Bloc("loopExit",
			Exit("mem")))

	if !doLoopRotation(fun) {
		t.Fatal("Loop rotation failed")
	}

	verifyRotatedCFG(fun, t)

	// one lives in loop latch and one lives in loop guard
	verifyNumValue(fun, t, OpLess64, 2)
}

// Loop contains multiple exits, and every loop exit block contians at least one
// use that uses loop phi.
//
//	if 0 < 10 {
//		i := 0
//		do {
//			if i == 1 {
//				use1 = i * 10
//				return
//			}
//			i++
//		} while i < 10
//	}
//	use2  = i * 10
func TestMultiExitLoopUses(t *testing.T) {
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
			Valu("use1", OpMul64, c.config.Types.Int64, 0, nil, "i", "ten"),
			Exit("mem")),
		Bloc("loopExit",
			Valu("use2", OpMul64, c.config.Types.Int64, 0, nil, "i", "ten"),
			Exit("mem")))

	if !doLoopRotationWithLCSSSA(fun) {
		t.Fatal("Loop rotation failed")
	}

	verifyRotatedCFG(fun, t)

	// one lives in loop latch and one lives in loop guard
	verifyNumValue(fun, t, OpLess64, 2)
	verifyNumValue(fun, t, OpPhi, 1 /*var i*/ +2 /*proxy phi*/)
}

// Even harder, Values defined in loop header are used everywhere.
func TestMultiExitLoopUsesEverywhere(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("loopEntry",
		Bloc("loopEntry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("zero", OpConst64, c.config.Types.Int64, 0, nil),
			Valu("one", OpConst64, c.config.Types.Int64, 1, nil),
			Valu("ten", OpConst64, c.config.Types.Int64, 10, nil),
			Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
			Goto("loopHeader")),
		Bloc("loopHeader",
			Valu("i", OpPhi, c.config.Types.Int64, 0, nil, "zero", "inc"),
			Valu("addr", OpAddr, c.config.Types.Int64.PtrTo(), 0, nil, "sb"),
			Valu("load", OpLoad, c.config.Types.Int64, 0, nil, "addr", "mem"),
			Valu("cmp", OpLess64, c.config.Types.Bool, 0, nil, "i", "load"),
			If("cmp", "loopBody", "loopExit")),
		Bloc("loopBody",
			Valu("use3", OpMul64, c.config.Types.Int64, 0, nil, "i", "load"),
			Valu("cmp2", OpEq64, c.config.Types.Bool, 0, nil, "i", "one"),
			If("cmp2", "loopExit1", "loopLatch")),
		Bloc("loopLatch",
			Valu("inc", OpAdd64, c.config.Types.Int64, 0, nil, "one", "i"),
			Goto("loopHeader")),
		Bloc("loopExit1",
			Valu("use1", OpMul64, c.config.Types.Int64, 0, nil, "i", "load"),
			Exit("mem")),
		Bloc("loopExit",
			Valu("use2", OpMul64, c.config.Types.Int64, 0, nil, "i", "load"),
			Exit("mem")))

	if !doLoopRotationWithLCSSSA(fun) {
		t.Fatal("Loop rotation failed")
	}

	verifyRotatedCFG(fun, t)

	// one lives in loop latch and one lives in loop guard
	verifyNumValue(fun, t, OpLess64, 2)
	verifyNumValue(fun, t, OpLoad, 2)
	numOfPhi := 2 /*two proxy phi in exit1*/ + 2 /*two proxy phi in exit*/ +
		2 /*i and inserted phi for load*/
	verifyNumValue(fun, t, OpPhi, numOfPhi)
}

// Rotation the Loop inclduing nesting children
func TestNestLoopRotation(t *testing.T) {
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
			If("cmp", "loopHeader2", "loopExit")),
		Bloc("loopHeader2",
			Valu("k", OpPhi, c.config.Types.Int64, 0, nil, "i", "inc2"),
			Valu("cmp2", OpEq64, c.config.Types.Bool, 0, nil, "k", "one"),
			If("cmp2", "loopLatch2", "loopLatch")),
		Bloc("loopLatch2",
			Valu("inc2", OpAdd64, c.config.Types.Int64, 0, nil, "one", "k"),
			Goto("loopHeader2")),
		Bloc("loopLatch",
			Valu("inc", OpAdd64, c.config.Types.Int64, 0, nil, "one", "i"),
			Goto("loopHeader")),
		Bloc("loopExit",
			Exit("mem")))

	if !doLoopRotationWithLCSSSA(fun) {
		t.Fatal("Loop rotation failed")
	}
}

// Store is defined in loop header and used outside the loop indirectly.
func TestBadLoop(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("loopEntry",
		Bloc("loopEntry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("zero", OpConst64, c.config.Types.Int64, 0, nil),
			Valu("one", OpConst64, c.config.Types.Int64, 1, nil),
			Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
			Goto("loopHeader")),
		Bloc("loopHeader",
			Valu("i", OpPhi, c.config.Types.Int64, 0, nil, "zero", "inc"),
			Valu("addr", OpAddr, c.config.Types.Int64.PtrTo(), 0, nil, "sb"),
			Valu("store", OpStore, types.TypeMem, 0, nil, "addr", "one", "mem"),
			Valu("load", OpLoad, c.config.Types.Int64, 0, nil, "addr", "store"),
			Valu("cmp", OpLess64, c.config.Types.Bool, 0, nil, "i", "load"),
			If("cmp", "loopLatch", "loopExit")),
		Bloc("loopLatch",
			Valu("inc", OpAdd64, c.config.Types.Int64, 0, nil, "load", "i"),
			Goto("loopHeader")),
		Bloc("loopExit",
			Valu("use", OpMul64, c.config.Types.Int64, 0, nil, "load", "one"),
			Exit("mem")))

	if doLoopRotationWithLCSSSA(fun) != false {
		t.Fatal("Loop rotation is expected to fail")
	}
}

// Loop def is non trivial because it excesses max depth
func TestBadLoop2(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("loopEntry",
		Bloc("loopEntry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("zero", OpConst64, c.config.Types.Int64, 0, nil),
			Valu("one", OpConst64, c.config.Types.Int64, 1, nil),
			Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
			Goto("loopHeader")),
		Bloc("loopHeader",
			Valu("i", OpPhi, c.config.Types.Int64, 0, nil, "zero", "inc"),
			Valu("addr", OpAddr, c.config.Types.Int64.PtrTo(), 0, nil, "sb"),
			Valu("load", OpLoad, c.config.Types.Int64, 0, nil, "addr", "mem"),
			Valu("depth5", OpAdd64, c.config.Types.Int64, 0, nil, "one", "load"),
			Valu("depth4", OpAdd64, c.config.Types.Int64, 0, nil, "one", "depth5"),
			Valu("depth3", OpAdd64, c.config.Types.Int64, 0, nil, "one", "depth4"),
			Valu("depth2", OpAdd64, c.config.Types.Int64, 0, nil, "one", "depth3"),
			Valu("cmp", OpLess64, c.config.Types.Bool, 0, nil, "i", "depth2"),
			If("cmp", "loopLatch", "loopExit")),
		Bloc("loopLatch",
			Valu("inc", OpAdd64, c.config.Types.Int64, 0, nil, "load", "i"),
			Goto("loopHeader")),
		Bloc("loopExit",
			Valu("use", OpMul64, c.config.Types.Int64, 0, nil, "load", "one"),
			Exit("mem")))

	if doLoopRotationWithLCSSSA(fun) != false {
		t.Fatal("Loop rotation is expected to fail")
	}
}

// Loop header has multiple entries
func TestBadLoop3(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("loopEntry1",
		Bloc("loopEntry1",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("zero", OpConst64, c.config.Types.Int64, 0, nil),
			Valu("one", OpConst64, c.config.Types.Int64, 1, nil),
			Valu("cmp", OpLess64, c.config.Types.Bool, 0, nil, "zero", "one"),
			If("cmp", "loopHeader", "loopEntry2")),
		Bloc("loopEntry2",
			Goto("loopHeader")),
		Bloc("loopHeader",
			Valu("i", OpPhi, c.config.Types.Int64, 0, nil, "zero", "one", "inc"),
			If("cmp", "loopLatch", "loopExit")),
		Bloc("loopLatch",
			Valu("inc", OpAdd64, c.config.Types.Int64, 0, nil, "one", "i"),
			Goto("loopHeader")),
		Bloc("loopExit",
			Exit("mem")))

	if doLoopRotationWithLCSSSA(fun) != false {
		t.Fatal("Loop rotation is expected to fail")
	}
}
