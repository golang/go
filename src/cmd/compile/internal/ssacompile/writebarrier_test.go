// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"testing"

	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/types"
)

func TestWriteBarrierStoreOrder(t *testing.T) {
	// Make sure writebarrier phase works even StoreWB ops are not in dependency order
	c := testConfig(t)
	ptrType := c.config.Types.BytePtr
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("start", ssaop.OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", ssaop.OpSB, c.config.Types.Uintptr, 0, nil),
			Valu("sp", ssaop.OpSP, c.config.Types.Uintptr, 0, nil),
			Valu("v", ssaop.OpConstNil, ptrType, 0, nil),
			Valu("addr1", ssaop.OpAddr, ptrType, 0, nil, "sb"),
			Valu("wb2", ssaop.OpStore, types.TypeMem, 0, ptrType, "addr1", "v", "wb1"),
			Valu("wb1", ssaop.OpStore, types.TypeMem, 0, ptrType, "addr1", "v", "start"), // wb1 and wb2 are out of order
			Goto("exit")),
		Bloc("exit",
			Exit("wb2")))

	CheckFunc(fun.f)
	writebarrier(fun.f)
	CheckFunc(fun.f)
}

func TestWriteBarrierPhi(t *testing.T) {
	// Make sure writebarrier phase works for single-block loop, where
	// a Phi op takes the store in the same block as argument.
	// See issue #19067.
	c := testConfig(t)
	ptrType := c.config.Types.BytePtr
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("start", ssaop.OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", ssaop.OpSB, c.config.Types.Uintptr, 0, nil),
			Valu("sp", ssaop.OpSP, c.config.Types.Uintptr, 0, nil),
			Goto("loop")),
		Bloc("loop",
			Valu("phi", ssaop.OpPhi, types.TypeMem, 0, nil, "start", "wb"),
			Valu("v", ssaop.OpConstNil, ptrType, 0, nil),
			Valu("addr", ssaop.OpAddr, ptrType, 0, nil, "sb"),
			Valu("wb", ssaop.OpStore, types.TypeMem, 0, ptrType, "addr", "v", "phi"), // has write barrier
			Goto("loop")))

	CheckFunc(fun.f)
	writebarrier(fun.f)
	CheckFunc(fun.f)
}
