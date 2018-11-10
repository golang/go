// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "testing"

func TestLiveControlOps(t *testing.T) {
	c := testConfig(t)
	f := Fun(c, "entry",
		Bloc("entry",
			Valu("mem", OpInitMem, TypeMem, 0, nil),
			Valu("x", OpAMD64MOVLconst, TypeInt8, 1, nil),
			Valu("y", OpAMD64MOVLconst, TypeInt8, 2, nil),
			Valu("a", OpAMD64TESTB, TypeFlags, 0, nil, "x", "y"),
			Valu("b", OpAMD64TESTB, TypeFlags, 0, nil, "y", "x"),
			Eq("a", "if", "exit"),
		),
		Bloc("if",
			Eq("b", "plain", "exit"),
		),
		Bloc("plain",
			Goto("exit"),
		),
		Bloc("exit",
			Exit("mem"),
		),
	)
	flagalloc(f.f)
	regalloc(f.f)
	checkFunc(f.f)
}

func TestSpillMove(t *testing.T) {
	// Test for issue 20472. We shouldn't move a spill out to exit blocks
	// if there is an exit block where the spill is dead but the pre-spill
	// value is live.
	c := testConfig(t)
	ptrType := &TypeImpl{Size_: 8, Ptr: true, Name: "testptr"} // dummy for testing
	arg1Aux := c.fe.Auto(TypeInt64)
	arg2Aux := c.fe.Auto(ptrType)
	f := Fun(c, "entry",
		Bloc("entry",
			Valu("mem", OpInitMem, TypeMem, 0, nil),
			Valu("x", OpArg, TypeInt64, 0, arg1Aux),
			Valu("p", OpArg, ptrType, 0, arg2Aux),
			Valu("a", OpAMD64TESTQ, TypeFlags, 0, nil, "x", "x"),
			Goto("loop1"),
		),
		Bloc("loop1",
			Valu("y", OpAMD64MULQ, TypeInt64, 0, nil, "x", "x"),
			Eq("a", "loop2", "exit1"),
		),
		Bloc("loop2",
			Eq("a", "loop1", "exit2"),
		),
		Bloc("exit1",
			// store before call, y is available in a register
			Valu("mem2", OpAMD64MOVQstore, TypeMem, 0, nil, "p", "y", "mem"),
			Valu("mem3", OpAMD64CALLstatic, TypeMem, 0, nil, "mem2"),
			Exit("mem3"),
		),
		Bloc("exit2",
			// store after call, y must be loaded from a spill location
			Valu("mem4", OpAMD64CALLstatic, TypeMem, 0, nil, "mem"),
			Valu("mem5", OpAMD64MOVQstore, TypeMem, 0, nil, "p", "y", "mem4"),
			Exit("mem5"),
		),
	)
	flagalloc(f.f)
	regalloc(f.f)
	checkFunc(f.f)
	// There should still be a spill in Loop1, and nowhere else.
	if numSpills(f.blocks["loop1"]) != 1 {
		t.Errorf("spill missing from loop1")
	}
	if numSpills(f.blocks["loop2"]) != 0 {
		t.Errorf("spill present in loop2")
	}
	if numSpills(f.blocks["exit1"]) != 0 {
		t.Errorf("spill present in exit1")
	}
	if numSpills(f.blocks["exit2"]) != 0 {
		t.Errorf("spill present in exit2")
	}

}

func numSpills(b *Block) int {
	n := 0
	for _, v := range b.Values {
		if v.Op == OpStoreReg {
			n++
		}
	}
	return n
}
