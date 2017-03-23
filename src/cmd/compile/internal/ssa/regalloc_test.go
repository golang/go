// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/internal/src"
	"testing"
)

func TestLiveControlOps(t *testing.T) {
	c := testConfig(t)
	f := c.Fun("entry",
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

// Test to make sure we don't push spills into loops.
// See issue #19595.
func TestSpillWithLoop(t *testing.T) {
	c := testConfig(t)
	f := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, TypeMem, 0, nil),
			Valu("ptr", OpArg, TypeInt64Ptr, 0, c.Frontend().Auto(src.NoXPos, TypeInt64)),
			Valu("cond", OpArg, TypeBool, 0, c.Frontend().Auto(src.NoXPos, TypeBool)),
			Valu("ld", OpAMD64MOVQload, TypeInt64, 0, nil, "ptr", "mem"), // this value needs a spill
			Goto("loop"),
		),
		Bloc("loop",
			Valu("memphi", OpPhi, TypeMem, 0, nil, "mem", "call"),
			Valu("call", OpAMD64CALLstatic, TypeMem, 0, nil, "memphi"),
			Valu("test", OpAMD64CMPBconst, TypeFlags, 0, nil, "cond"),
			Eq("test", "next", "exit"),
		),
		Bloc("next",
			Goto("loop"),
		),
		Bloc("exit",
			Valu("store", OpAMD64MOVQstore, TypeMem, 0, nil, "ptr", "ld", "call"),
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
