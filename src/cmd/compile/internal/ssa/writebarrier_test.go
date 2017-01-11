// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "testing"

func TestWriteBarrierStoreOrder(t *testing.T) {
	// Make sure writebarrier phase works even StoreWB ops are not in dependency order
	c := testConfig(t)
	ptrType := &TypeImpl{Size_: 8, Ptr: true, Name: "testptr"} // dummy for testing
	fun := Fun(c, "entry",
		Bloc("entry",
			Valu("start", OpInitMem, TypeMem, 0, nil),
			Valu("sb", OpSB, TypeInvalid, 0, nil),
			Valu("sp", OpSP, TypeInvalid, 0, nil),
			Valu("v", OpConstNil, ptrType, 0, nil),
			Valu("addr1", OpAddr, ptrType, 0, nil, "sb"),
			Valu("wb2", OpStoreWB, TypeMem, 8, nil, "addr1", "v", "wb1"),
			Valu("wb1", OpStoreWB, TypeMem, 8, nil, "addr1", "v", "start"), // wb1 and wb2 are out of order
			Goto("exit")),
		Bloc("exit",
			Exit("wb2")))

	CheckFunc(fun.f)
	writebarrier(fun.f)
	CheckFunc(fun.f)
}
