// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"testing"
)

func TestDeadStore(t *testing.T) {
	c := testConfig(t)
	ptrType := c.config.Types.BytePtr
	t.Logf("PTRTYPE %v", ptrType)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("start", OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
			Valu("v", OpConstBool, c.config.Types.Bool, 1, nil),
			Valu("addr1", OpAddr, ptrType, 0, nil, "sb"),
			Valu("addr2", OpAddr, ptrType, 0, nil, "sb"),
			Valu("addr3", OpAddr, ptrType, 0, nil, "sb"),
			Valu("zero1", OpZero, types.TypeMem, 1, c.config.Types.Bool, "addr3", "start"),
			Valu("store1", OpStore, types.TypeMem, 0, c.config.Types.Bool, "addr1", "v", "zero1"),
			Valu("store2", OpStore, types.TypeMem, 0, c.config.Types.Bool, "addr2", "v", "store1"),
			Valu("store3", OpStore, types.TypeMem, 0, c.config.Types.Bool, "addr1", "v", "store2"),
			Valu("store4", OpStore, types.TypeMem, 0, c.config.Types.Bool, "addr3", "v", "store3"),
			Goto("exit")),
		Bloc("exit",
			Exit("store3")))

	CheckFunc(fun.f)
	dse(fun.f)
	CheckFunc(fun.f)

	v1 := fun.values["store1"]
	if v1.Op != OpCopy {
		t.Errorf("dead store not removed")
	}

	v2 := fun.values["zero1"]
	if v2.Op != OpCopy {
		t.Errorf("dead store (zero) not removed")
	}
}
func TestDeadStorePhi(t *testing.T) {
	// make sure we don't get into an infinite loop with phi values.
	c := testConfig(t)
	ptrType := c.config.Types.BytePtr
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("start", OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
			Valu("v", OpConstBool, c.config.Types.Bool, 1, nil),
			Valu("addr", OpAddr, ptrType, 0, nil, "sb"),
			Goto("loop")),
		Bloc("loop",
			Valu("phi", OpPhi, types.TypeMem, 0, nil, "start", "store"),
			Valu("store", OpStore, types.TypeMem, 0, c.config.Types.Bool, "addr", "v", "phi"),
			If("v", "loop", "exit")),
		Bloc("exit",
			Exit("store")))

	CheckFunc(fun.f)
	dse(fun.f)
	CheckFunc(fun.f)
}

func TestDeadStoreTypes(t *testing.T) {
	// Make sure a narrow store can't shadow a wider one. We test an even
	// stronger restriction, that one store can't shadow another unless the
	// types of the address fields are identical (where identicalness is
	// decided by the CSE pass).
	c := testConfig(t)
	t1 := c.config.Types.UInt64.PtrTo()
	t2 := c.config.Types.UInt32.PtrTo()
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("start", OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
			Valu("v", OpConstBool, c.config.Types.Bool, 1, nil),
			Valu("addr1", OpAddr, t1, 0, nil, "sb"),
			Valu("addr2", OpAddr, t2, 0, nil, "sb"),
			Valu("store1", OpStore, types.TypeMem, 0, c.config.Types.Bool, "addr1", "v", "start"),
			Valu("store2", OpStore, types.TypeMem, 0, c.config.Types.Bool, "addr2", "v", "store1"),
			Goto("exit")),
		Bloc("exit",
			Exit("store2")))

	CheckFunc(fun.f)
	cse(fun.f)
	dse(fun.f)
	CheckFunc(fun.f)

	v := fun.values["store1"]
	if v.Op == OpCopy {
		t.Errorf("store %s incorrectly removed", v)
	}
}

func TestDeadStoreUnsafe(t *testing.T) {
	// Make sure a narrow store can't shadow a wider one. The test above
	// covers the case of two different types, but unsafe pointer casting
	// can get to a point where the size is changed but type unchanged.
	c := testConfig(t)
	ptrType := c.config.Types.UInt64.PtrTo()
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("start", OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
			Valu("v", OpConstBool, c.config.Types.Bool, 1, nil),
			Valu("addr1", OpAddr, ptrType, 0, nil, "sb"),
			Valu("store1", OpStore, types.TypeMem, 0, c.config.Types.Int64, "addr1", "v", "start"), // store 8 bytes
			Valu("store2", OpStore, types.TypeMem, 0, c.config.Types.Bool, "addr1", "v", "store1"), // store 1 byte
			Goto("exit")),
		Bloc("exit",
			Exit("store2")))

	CheckFunc(fun.f)
	cse(fun.f)
	dse(fun.f)
	CheckFunc(fun.f)

	v := fun.values["store1"]
	if v.Op == OpCopy {
		t.Errorf("store %s incorrectly removed", v)
	}
}
