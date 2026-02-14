// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"cmd/internal/obj/x86"
	"fmt"
	"testing"
)

func TestLiveControlOps(t *testing.T) {
	c := testConfig(t)
	f := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("x", OpAMD64MOVLconst, c.config.Types.Int8, 1, nil),
			Valu("y", OpAMD64MOVLconst, c.config.Types.Int8, 2, nil),
			Valu("a", OpAMD64TESTB, types.TypeFlags, 0, nil, "x", "y"),
			Valu("b", OpAMD64TESTB, types.TypeFlags, 0, nil, "y", "x"),
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

// Test to make sure G register is never reloaded from spill (spill of G is okay)
// See #25504
func TestNoGetgLoadReg(t *testing.T) {
	/*
		Original:
		func fff3(i int) *g {
			gee := getg()
			if i == 0 {
				fff()
			}
			return gee // here
		}
	*/
	c := testConfigARM64(t)
	f := c.Fun("b1",
		Bloc("b1",
			Valu("v1", OpInitMem, types.TypeMem, 0, nil),
			Valu("v6", OpArg, c.config.Types.Int64, 0, c.Temp(c.config.Types.Int64)),
			Valu("v8", OpGetG, c.config.Types.Int64.PtrTo(), 0, nil, "v1"),
			Valu("v11", OpARM64CMPconst, types.TypeFlags, 0, nil, "v6"),
			Eq("v11", "b2", "b4"),
		),
		Bloc("b4",
			Goto("b3"),
		),
		Bloc("b3",
			Valu("v14", OpPhi, types.TypeMem, 0, nil, "v1", "v12"),
			Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
			Valu("v16", OpARM64MOVDstore, types.TypeMem, 0, nil, "v8", "sb", "v14"),
			Exit("v16"),
		),
		Bloc("b2",
			Valu("v12", OpARM64CALLstatic, types.TypeMem, 0, AuxCallLSym("_"), "v1"),
			Goto("b3"),
		),
	)
	regalloc(f.f)
	checkFunc(f.f)
	// Double-check that we never restore to the G register. Regalloc should catch it, but check again anyway.
	r := f.f.RegAlloc
	for _, b := range f.blocks {
		for _, v := range b.Values {
			if v.Op == OpLoadReg && r[v.ID].String() == "g" {
				t.Errorf("Saw OpLoadReg targeting g register: %s", v.LongString())
			}
		}
	}
}

// Test to make sure we don't push spills into loops.
// See issue #19595.
func TestSpillWithLoop(t *testing.T) {
	c := testConfig(t)
	f := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("ptr", OpArg, c.config.Types.Int64.PtrTo(), 0, c.Temp(c.config.Types.Int64)),
			Valu("cond", OpArg, c.config.Types.Bool, 0, c.Temp(c.config.Types.Bool)),
			Valu("ld", OpAMD64MOVQload, c.config.Types.Int64, 0, nil, "ptr", "mem"), // this value needs a spill
			Goto("loop"),
		),
		Bloc("loop",
			Valu("memphi", OpPhi, types.TypeMem, 0, nil, "mem", "call"),
			Valu("call", OpAMD64CALLstatic, types.TypeMem, 0, AuxCallLSym("_"), "memphi"),
			Valu("test", OpAMD64CMPBconst, types.TypeFlags, 0, nil, "cond"),
			Eq("test", "next", "exit"),
		),
		Bloc("next",
			Goto("loop"),
		),
		Bloc("exit",
			Valu("store", OpAMD64MOVQstore, types.TypeMem, 0, nil, "ptr", "ld", "call"),
			Exit("store"),
		),
	)
	regalloc(f.f)
	checkFunc(f.f)
	for _, v := range f.blocks["loop"].Values {
		if v.Op == OpStoreReg {
			t.Errorf("spill inside loop %s", v.LongString())
		}
	}
}

func TestSpillMove1(t *testing.T) {
	c := testConfig(t)
	f := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("x", OpArg, c.config.Types.Int64, 0, c.Temp(c.config.Types.Int64)),
			Valu("p", OpArg, c.config.Types.Int64.PtrTo(), 0, c.Temp(c.config.Types.Int64.PtrTo())),
			Valu("a", OpAMD64TESTQ, types.TypeFlags, 0, nil, "x", "x"),
			Goto("loop1"),
		),
		Bloc("loop1",
			Valu("y", OpAMD64MULQ, c.config.Types.Int64, 0, nil, "x", "x"),
			Eq("a", "loop2", "exit1"),
		),
		Bloc("loop2",
			Eq("a", "loop1", "exit2"),
		),
		Bloc("exit1",
			// store before call, y is available in a register
			Valu("mem2", OpAMD64MOVQstore, types.TypeMem, 0, nil, "p", "y", "mem"),
			Valu("mem3", OpAMD64CALLstatic, types.TypeMem, 0, AuxCallLSym("_"), "mem2"),
			Exit("mem3"),
		),
		Bloc("exit2",
			// store after call, y must be loaded from a spill location
			Valu("mem4", OpAMD64CALLstatic, types.TypeMem, 0, AuxCallLSym("_"), "mem"),
			Valu("mem5", OpAMD64MOVQstore, types.TypeMem, 0, nil, "p", "y", "mem4"),
			Exit("mem5"),
		),
	)
	flagalloc(f.f)
	regalloc(f.f)
	checkFunc(f.f)
	// Spill should be moved to exit2.
	if numSpills(f.blocks["loop1"]) != 0 {
		t.Errorf("spill present from loop1")
	}
	if numSpills(f.blocks["loop2"]) != 0 {
		t.Errorf("spill present in loop2")
	}
	if numSpills(f.blocks["exit1"]) != 0 {
		t.Errorf("spill present in exit1")
	}
	if numSpills(f.blocks["exit2"]) != 1 {
		t.Errorf("spill missing in exit2")
	}

}

func TestSpillMove2(t *testing.T) {
	c := testConfig(t)
	f := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("x", OpArg, c.config.Types.Int64, 0, c.Temp(c.config.Types.Int64)),
			Valu("p", OpArg, c.config.Types.Int64.PtrTo(), 0, c.Temp(c.config.Types.Int64.PtrTo())),
			Valu("a", OpAMD64TESTQ, types.TypeFlags, 0, nil, "x", "x"),
			Goto("loop1"),
		),
		Bloc("loop1",
			Valu("y", OpAMD64MULQ, c.config.Types.Int64, 0, nil, "x", "x"),
			Eq("a", "loop2", "exit1"),
		),
		Bloc("loop2",
			Eq("a", "loop1", "exit2"),
		),
		Bloc("exit1",
			// store after call, y must be loaded from a spill location
			Valu("mem2", OpAMD64CALLstatic, types.TypeMem, 0, AuxCallLSym("_"), "mem"),
			Valu("mem3", OpAMD64MOVQstore, types.TypeMem, 0, nil, "p", "y", "mem2"),
			Exit("mem3"),
		),
		Bloc("exit2",
			// store after call, y must be loaded from a spill location
			Valu("mem4", OpAMD64CALLstatic, types.TypeMem, 0, AuxCallLSym("_"), "mem"),
			Valu("mem5", OpAMD64MOVQstore, types.TypeMem, 0, nil, "p", "y", "mem4"),
			Exit("mem5"),
		),
	)
	flagalloc(f.f)
	regalloc(f.f)
	checkFunc(f.f)
	// There should be a spill in loop1, and nowhere else.
	// TODO: resurrect moving spills out of loops? We could put spills at the start of both exit1 and exit2.
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

func TestClobbersArg0(t *testing.T) {
	c := testConfig(t)
	f := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("ptr", OpArg, c.config.Types.Int64.PtrTo(), 0, c.Temp(c.config.Types.Int64.PtrTo())),
			Valu("dst", OpArg, c.config.Types.Int64.PtrTo().PtrTo(), 0, c.Temp(c.config.Types.Int64.PtrTo().PtrTo())),
			Valu("zero", OpAMD64LoweredZeroLoop, types.TypeMem, 256, nil, "ptr", "mem"),
			Valu("store", OpAMD64MOVQstore, types.TypeMem, 0, nil, "dst", "ptr", "zero"),
			Exit("store")))
	flagalloc(f.f)
	regalloc(f.f)
	checkFunc(f.f)
	// LoweredZeroLoop clobbers its argument, so there must be a copy of "ptr" somewhere
	// so we still have that value available at "store".
	if n := numCopies(f.blocks["entry"]); n != 1 {
		fmt.Printf("%s\n", f.f.String())
		t.Errorf("got %d copies, want 1", n)
	}
}

func TestClobbersArg1(t *testing.T) {
	c := testConfig(t)
	f := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("src", OpArg, c.config.Types.Int64.PtrTo(), 0, c.Temp(c.config.Types.Int64.PtrTo())),
			Valu("dst", OpArg, c.config.Types.Int64.PtrTo(), 0, c.Temp(c.config.Types.Int64.PtrTo())),
			Valu("use1", OpArg, c.config.Types.Int64.PtrTo().PtrTo(), 0, c.Temp(c.config.Types.Int64.PtrTo().PtrTo())),
			Valu("use2", OpArg, c.config.Types.Int64.PtrTo().PtrTo(), 0, c.Temp(c.config.Types.Int64.PtrTo().PtrTo())),
			Valu("move", OpAMD64LoweredMoveLoop, types.TypeMem, 256, nil, "dst", "src", "mem"),
			Valu("store1", OpAMD64MOVQstore, types.TypeMem, 0, nil, "use1", "src", "move"),
			Valu("store2", OpAMD64MOVQstore, types.TypeMem, 0, nil, "use2", "dst", "store1"),
			Exit("store2")))
	flagalloc(f.f)
	regalloc(f.f)
	checkFunc(f.f)
	// LoweredMoveLoop clobbers its arguments, so there must be a copy of "src" and "dst" somewhere
	// so we still have that value available at the stores.
	if n := numCopies(f.blocks["entry"]); n != 2 {
		fmt.Printf("%s\n", f.f.String())
		t.Errorf("got %d copies, want 2", n)
	}
}

func TestNoRematerializeDeadConstant(t *testing.T) {
	c := testConfigARM64(t)
	f := c.Fun("b1",
		Bloc("b1",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("addr", OpArg, c.config.Types.Int32.PtrTo(), 0, c.Temp(c.config.Types.Int32.PtrTo())),
			Valu("const", OpARM64MOVDconst, c.config.Types.Int32, -1, nil), // Original constant
			Valu("cmp", OpARM64CMPconst, types.TypeFlags, 0, nil, "const"),
			Goto("b2"),
		),
		Bloc("b2",
			Valu("phi_mem", OpPhi, types.TypeMem, 0, nil, "mem", "callmem"),
			Eq("cmp", "b6", "b3"),
		),
		Bloc("b3",
			Valu("call", OpARM64CALLstatic, types.TypeMem, 0, AuxCallLSym("_"), "phi_mem"),
			Valu("callmem", OpSelectN, types.TypeMem, 0, nil, "call"),
			Eq("cmp", "b5", "b4"),
		),
		Bloc("b4", // A block where we don't really need to rematerialize the constant -1
			Goto("b2"),
		),
		Bloc("b5",
			Valu("user", OpAMD64MOVQstore, types.TypeMem, 0, nil, "addr", "const", "callmem"),
			Exit("user"),
		),
		Bloc("b6",
			Exit("phi_mem"),
		),
	)

	regalloc(f.f)
	checkFunc(f.f)

	// Check that in block b4, there's no dead rematerialization of the constant -1
	for _, v := range f.blocks["b4"].Values {
		if v.Op == OpARM64MOVDconst && v.AuxInt == -1 {
			t.Errorf("constant -1 rematerialized in loop block b4: %s", v.LongString())
		}
	}
}

func numSpills(b *Block) int {
	return numOps(b, OpStoreReg)
}
func numCopies(b *Block) int {
	return numOps(b, OpCopy)
}
func numOps(b *Block, op Op) int {
	n := 0
	for _, v := range b.Values {
		if v.Op == op {
			n++
		}
	}
	return n
}

func TestRematerializeableRegCompatible(t *testing.T) {
	c := testConfig(t)
	f := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("x", OpAMD64MOVLconst, c.config.Types.Int32, 1, nil),
			Valu("a", OpAMD64POR, c.config.Types.Float32, 0, nil, "x", "x"),
			Valu("res", OpMakeResult, types.NewResults([]*types.Type{c.config.Types.Float32, types.TypeMem}), 0, nil, "a", "mem"),
			Ret("res"),
		),
	)
	regalloc(f.f)
	checkFunc(f.f)
	moveFound := false
	for _, v := range f.f.Blocks[0].Values {
		if v.Op == OpCopy && x86.REG_X0 <= v.Reg() && v.Reg() <= x86.REG_X31 {
			moveFound = true
		}
	}
	if !moveFound {
		t.Errorf("Expects an Copy to be issued, but got: %+v", f.f)
	}
}

func TestPreload(t *testing.T) {
	c := testConfig(t)
	// amd64 has 13 general registers. We use 1 for ptr and 12 for x0-11.
	// They all contain live values at the end of the entry block.
	f := c.Fun("entry",
		Bloc("entry",
			Valu("ptr", OpArgIntReg, c.config.Types.Int8.PtrTo(), 0, &AuxNameOffset{Name: c.Temp(c.config.Types.Int8.PtrTo()), Offset: 0}),
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("x0", OpAMD64MOVBload, c.config.Types.Int8, 0, nil, "ptr", "mem"),
			Valu("x1", OpAMD64MOVBload, c.config.Types.Int8, 1, nil, "ptr", "mem"),
			Valu("x2", OpAMD64MOVBload, c.config.Types.Int8, 2, nil, "ptr", "mem"),
			Valu("x3", OpAMD64MOVBload, c.config.Types.Int8, 3, nil, "ptr", "mem"),
			Valu("x4", OpAMD64MOVBload, c.config.Types.Int8, 4, nil, "ptr", "mem"),
			Valu("x5", OpAMD64MOVBload, c.config.Types.Int8, 5, nil, "ptr", "mem"),
			Valu("x6", OpAMD64MOVBload, c.config.Types.Int8, 6, nil, "ptr", "mem"),
			Valu("x7", OpAMD64MOVBload, c.config.Types.Int8, 7, nil, "ptr", "mem"),
			Valu("x8", OpAMD64MOVBload, c.config.Types.Int8, 8, nil, "ptr", "mem"),
			Valu("x9", OpAMD64MOVBload, c.config.Types.Int8, 9, nil, "ptr", "mem"),
			Valu("x10", OpAMD64MOVBload, c.config.Types.Int8, 10, nil, "ptr", "mem"),
			Valu("x11", OpAMD64MOVBload, c.config.Types.Int8, 11, nil, "ptr", "mem"),
			Valu("init", OpAMD64MOVQconst, c.config.Types.Int64, 0, nil),
			Goto("loopHead"),
		),
		Bloc("loopHead",
			Valu("phi", OpPhi, c.config.Types.Int64, 0, nil, "init", "next"),
			Valu("test", OpAMD64CMPQconst, types.TypeFlags, 10, nil, "phi"),
			Lt("test", "loopBody", "exit"),
		),
		Bloc("loopBody",
			Valu("next", OpAMD64ADDQconst, c.config.Types.Int64, 1, nil, "phi"),
			Goto("loopHead"),
		),
		Bloc("exit",
			Valu("m0", OpAMD64MOVBstore, types.TypeMem, 0, nil, "ptr", "x0", "mem"),
			Valu("m1", OpAMD64MOVBstore, types.TypeMem, 1, nil, "ptr", "x1", "m0"),
			Valu("m2", OpAMD64MOVBstore, types.TypeMem, 2, nil, "ptr", "x2", "m1"),
			Valu("m3", OpAMD64MOVBstore, types.TypeMem, 3, nil, "ptr", "x3", "m2"),
			Valu("m4", OpAMD64MOVBstore, types.TypeMem, 4, nil, "ptr", "x4", "m3"),
			Valu("m5", OpAMD64MOVBstore, types.TypeMem, 5, nil, "ptr", "x5", "m4"),
			Valu("m6", OpAMD64MOVBstore, types.TypeMem, 6, nil, "ptr", "x6", "m5"),
			Valu("m7", OpAMD64MOVBstore, types.TypeMem, 7, nil, "ptr", "x7", "m6"),
			Valu("m8", OpAMD64MOVBstore, types.TypeMem, 8, nil, "ptr", "x8", "m7"),
			Valu("m9", OpAMD64MOVBstore, types.TypeMem, 9, nil, "ptr", "x9", "m8"),
			Valu("m10", OpAMD64MOVBstore, types.TypeMem, 10, nil, "ptr", "x10", "m9"),
			Valu("m11", OpAMD64MOVBstore, types.TypeMem, 11, nil, "ptr", "x11", "m10"),
			Ret("m11"),
		),
	)
	f.f.Blocks[1].Likely = BranchLikely
	regalloc(f.f)
	checkFunc(f.f)

	v := f.values["phi"]
	loc := f.f.RegAlloc[v.ID]
	if _, ok := loc.(*Register); !ok {
		t.Errorf("Expects to use a register for phi, but got: %s\n%v", loc, f.f)
	}
}

// TestStartRegsDrop tests dropping physical register bits from startRegs preventing
// unnecessary OpLoadReg on edges, such as edge right->merge in this example.
//
//	entry -> left (no pressure)  -> merge (startRegs updated) -> exit
//	      -> right (pressure)   ->
func TestStartRegsDrop(t *testing.T) {
	c := testConfig(t)

	f := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("ptr", OpArg, c.config.Types.Int64.PtrTo(), 0, c.Temp(c.config.Types.Int64)),
			// Non-phi values that will be originally in registers and get into startRegs[merge]
			Valu("u0", OpAMD64MOVQload, c.config.Types.Int64, 100, nil, "ptr", "mem"),
			Valu("u1", OpAMD64MOVQload, c.config.Types.Int64, 108, nil, "ptr", "mem"),
			Valu("u2", OpAMD64MOVQload, c.config.Types.Int64, 116, nil, "ptr", "mem"),
			Valu("u3", OpAMD64MOVQload, c.config.Types.Int64, 124, nil, "ptr", "mem"),
			Valu("u4", OpAMD64MOVQload, c.config.Types.Int64, 132, nil, "ptr", "mem"),
			Valu("u5", OpAMD64MOVQload, c.config.Types.Int64, 140, nil, "ptr", "mem"),
			Valu("u6", OpAMD64MOVQload, c.config.Types.Int64, 148, nil, "ptr", "mem"),
			Valu("u7", OpAMD64MOVQload, c.config.Types.Int64, 156, nil, "ptr", "mem"),
			// Some phi value
			Valu("v0", OpAMD64MOVQload, c.config.Types.Int64, 0, nil, "ptr", "mem"),
			// Branch
			Valu("cond", OpAMD64MOVLconst, c.config.Types.Int32, 1, nil),
			Valu("test", OpAMD64TESTL, types.TypeFlags, 0, nil, "cond", "cond"),
			Eq("test", "left", "right"),
		),
		Bloc("left",
			// No register pressure - values stay in their original registers
			// This is the "primary" predecessor of merge block (less spill live)
			Goto("merge"),
		),
		Bloc("right",
			// Create some register pressure. We want to make difference in this block's endRegs,
			// so that at shuffle stage, the updated startRegs of merge block will be measurable.
			Valu("r0", OpAMD64MOVQload, c.config.Types.Int64, 200, nil, "ptr", "mem"),
			Valu("r1", OpAMD64MOVQload, c.config.Types.Int64, 208, nil, "ptr", "mem"),
			Valu("r2", OpAMD64MOVQload, c.config.Types.Int64, 216, nil, "ptr", "mem"),
			Valu("r3", OpAMD64MOVQload, c.config.Types.Int64, 224, nil, "ptr", "mem"),
			Valu("r4", OpAMD64MOVQload, c.config.Types.Int64, 232, nil, "ptr", "mem"),
			Valu("r5", OpAMD64MOVQload, c.config.Types.Int64, 240, nil, "ptr", "mem"),
			Valu("r6", OpAMD64MOVQload, c.config.Types.Int64, 248, nil, "ptr", "mem"),
			Valu("r7", OpAMD64MOVQload, c.config.Types.Int64, 256, nil, "ptr", "mem"),
			Valu("r8", OpAMD64MOVQload, c.config.Types.Int64, 264, nil, "ptr", "mem"),
			Valu("r9", OpAMD64MOVQload, c.config.Types.Int64, 272, nil, "ptr", "mem"),
			Valu("r10", OpAMD64MOVQload, c.config.Types.Int64, 280, nil, "ptr", "mem"),
			Valu("r11", OpAMD64MOVQload, c.config.Types.Int64, 288, nil, "ptr", "mem"),
			Valu("r12", OpAMD64MOVQload, c.config.Types.Int64, 296, nil, "ptr", "mem"),
			Valu("r13", OpAMD64MOVQload, c.config.Types.Int64, 304, nil, "ptr", "mem"),
			Valu("r14", OpAMD64MOVQload, c.config.Types.Int64, 312, nil, "ptr", "mem"),
			Valu("r15", OpAMD64MOVQload, c.config.Types.Int64, 320, nil, "ptr", "mem"),
			Valu("sum0", OpAMD64ADDQ, c.config.Types.Int64, 0, nil, "r0", "r1"),
			Valu("sum1", OpAMD64ADDQ, c.config.Types.Int64, 0, nil, "r2", "r3"),
			Valu("sum2", OpAMD64ADDQ, c.config.Types.Int64, 0, nil, "r4", "r5"),
			Valu("sum3", OpAMD64ADDQ, c.config.Types.Int64, 0, nil, "r6", "r7"),
			Valu("sum4", OpAMD64ADDQ, c.config.Types.Int64, 0, nil, "r8", "r9"),
			Valu("sum5", OpAMD64ADDQ, c.config.Types.Int64, 0, nil, "r10", "r11"),
			Valu("sum6", OpAMD64ADDQ, c.config.Types.Int64, 0, nil, "r12", "r13"),
			Valu("sum7", OpAMD64ADDQ, c.config.Types.Int64, 0, nil, "r14", "r15"),
			Valu("store0", OpAMD64MOVQstore, types.TypeMem, 400, nil, "ptr", "sum0", "mem"),
			Valu("store1", OpAMD64MOVQstore, types.TypeMem, 408, nil, "ptr", "sum1", "store0"),
			Valu("store2", OpAMD64MOVQstore, types.TypeMem, 416, nil, "ptr", "sum2", "store1"),
			Valu("store3", OpAMD64MOVQstore, types.TypeMem, 424, nil, "ptr", "sum3", "store2"),
			Valu("store4", OpAMD64MOVQstore, types.TypeMem, 432, nil, "ptr", "sum4", "store3"),
			Valu("store5", OpAMD64MOVQstore, types.TypeMem, 440, nil, "ptr", "sum5", "store4"),
			Valu("store6", OpAMD64MOVQstore, types.TypeMem, 448, nil, "ptr", "sum6", "store5"),
			Valu("store7", OpAMD64MOVQstore, types.TypeMem, 456, nil, "ptr", "sum7", "store6"),
			Goto("merge"),
		),
		Bloc("merge",
			// One phi (for v0)
			Valu("p0", OpPhi, c.config.Types.Int64, 0, nil, "v0", "v0"),
			// New values to evict the most distant uses in the end of this block
			Valu("n0", OpAMD64MOVQload, c.config.Types.Int64, 500, nil, "ptr", "mem"),
			Valu("n1", OpAMD64MOVQload, c.config.Types.Int64, 508, nil, "ptr", "mem"),
			Valu("n2", OpAMD64MOVQload, c.config.Types.Int64, 516, nil, "ptr", "mem"),
			Valu("n3", OpAMD64MOVQload, c.config.Types.Int64, 524, nil, "ptr", "mem"),
			Valu("n4", OpAMD64MOVQload, c.config.Types.Int64, 532, nil, "ptr", "mem"),
			Valu("n5", OpAMD64MOVQload, c.config.Types.Int64, 540, nil, "ptr", "mem"),
			Valu("n6", OpAMD64MOVQload, c.config.Types.Int64, 548, nil, "ptr", "mem"),
			Valu("n7", OpAMD64MOVQload, c.config.Types.Int64, 556, nil, "ptr", "mem"),
			Valu("n8", OpAMD64MOVQload, c.config.Types.Int64, 564, nil, "ptr", "mem"),
			Valu("n9", OpAMD64MOVQload, c.config.Types.Int64, 572, nil, "ptr", "mem"),
			Valu("n10", OpAMD64MOVQload, c.config.Types.Int64, 580, nil, "ptr", "mem"),
			Valu("n11", OpAMD64MOVQload, c.config.Types.Int64, 588, nil, "ptr", "mem"),
			Valu("n12", OpAMD64MOVQload, c.config.Types.Int64, 596, nil, "ptr", "mem"),
			Valu("n13", OpAMD64MOVQload, c.config.Types.Int64, 604, nil, "ptr", "mem"),
			// Normal uses before we start to evict / clean up startRegs mask
			Valu("a0", OpAMD64ADDQ, c.config.Types.Int64, 0, nil, "n0", "n1"),
			Valu("a1", OpAMD64ADDQ, c.config.Types.Int64, 0, nil, "n2", "n3"),
			Valu("a2", OpAMD64ADDQ, c.config.Types.Int64, 0, nil, "n4", "n5"),
			Valu("a3", OpAMD64ADDQ, c.config.Types.Int64, 0, nil, "n6", "n7"),
			Valu("a4", OpAMD64ADDQ, c.config.Types.Int64, 0, nil, "n8", "n9"),
			Valu("a5", OpAMD64ADDQ, c.config.Types.Int64, 0, nil, "n10", "n11"),
			Valu("a6", OpAMD64ADDQ, c.config.Types.Int64, 0, nil, "n12", "n13"),
			Valu("s0", OpAMD64MOVQstore, types.TypeMem, 0, nil, "ptr", "a0", "mem"),
			Valu("s1", OpAMD64MOVQstore, types.TypeMem, 8, nil, "ptr", "a1", "s0"),
			Valu("s2", OpAMD64MOVQstore, types.TypeMem, 16, nil, "ptr", "a2", "s1"),
			Valu("s3", OpAMD64MOVQstore, types.TypeMem, 24, nil, "ptr", "a3", "s2"),
			Valu("s4", OpAMD64MOVQstore, types.TypeMem, 32, nil, "ptr", "a4", "s3"),
			Valu("s5", OpAMD64MOVQstore, types.TypeMem, 40, nil, "ptr", "a5", "s4"),
			Valu("s6", OpAMD64MOVQstore, types.TypeMem, 48, nil, "ptr", "a6", "s5"),
			// The distant uses - to be evicted and cleaned up from the startRegs mask
			Valu("t0", OpAMD64MOVQstore, types.TypeMem, 100, nil, "ptr", "u0", "s6"),
			Valu("t1", OpAMD64MOVQstore, types.TypeMem, 108, nil, "ptr", "u1", "t0"),
			Valu("t2", OpAMD64MOVQstore, types.TypeMem, 116, nil, "ptr", "u2", "t1"),
			Valu("t3", OpAMD64MOVQstore, types.TypeMem, 124, nil, "ptr", "u3", "t2"),
			Valu("t4", OpAMD64MOVQstore, types.TypeMem, 132, nil, "ptr", "u4", "t3"),
			Valu("t5", OpAMD64MOVQstore, types.TypeMem, 140, nil, "ptr", "u5", "t4"),
			Valu("t6", OpAMD64MOVQstore, types.TypeMem, 148, nil, "ptr", "u6", "t5"),
			Valu("t7", OpAMD64MOVQstore, types.TypeMem, 156, nil, "ptr", "u7", "t6"),
			Exit("t7"),
		),
	)

	regalloc(f.f)
	checkFunc(f.f)

	leftLoadCount := numOps(f.blocks["left"], OpLoadReg)
	rightLoadCount := numOps(f.blocks["right"], OpLoadReg)
	t.Logf("OpLoadReg count in left: %d, right: %d", leftLoadCount, rightLoadCount)

	// Without startRegs mask cleanup we would have some dead LoadRegs added by shuffle
	if rightLoadCount > 8 {
		t.Errorf("expected <= 8 OpLoadReg in right block (got %d)", rightLoadCount)
	}
}
