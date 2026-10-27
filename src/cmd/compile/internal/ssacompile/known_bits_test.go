// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"testing"

	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/types"
)

func TestRshCopiesSignBitEverywhere(t *testing.T) {
	c := testConfig(t)

	f := c.Fun("entry",
		Bloc("entry",
			Valu("mem", ssaop.OpInitMem, types.TypeMem, 0, nil),
			Valu("x", ssaop.OpArg, c.config.Types.Int64, 0, nil),
			Valu("y", ssaop.OpArg, c.config.Types.Int64, 1, nil),
			Valu("ptr", ssaop.OpArg, c.config.Types.BytePtr, 2, nil),
			Valu("signBit", ssaop.OpConst64, c.config.Types.Int64, -1<<63, nil),
			Valu("xWithSignBitSet", ssaop.OpOr64, c.config.Types.Int64, 0, nil, "x", "signBit"),
			Valu("oneHundredTwentyEight", ssaop.OpConst64, c.config.Types.Int64, 128, nil),
			Valu("y", ssaop.OpOr64, c.config.Types.Int64, 0, nil, "y", "oneHundredTwentyEight"),
			Valu("rsh", ssaop.OpRsh64x64, c.config.Types.Int64, 0, nil, "xWithSignBitSet", "y"),
			Valu("one", ssaop.OpConst64, c.config.Types.Int64, 1, nil),
			Valu("and", ssaop.OpAnd64, c.config.Types.Int64, 0, nil, "rsh", "one"),
			Valu("mem", ssaop.OpStore, types.TypeMem, 0, nil, "ptr", "and", "mem"),
			Exit("mem")))

	CheckFunc(f.f)
	ssa.KnownBits(f.f)
	CheckFunc(f.f)

	if len(f.f.Blocks) != 1 {
		t.Fatalf("expected 1 block, got %d", len(f.f.Blocks))
	}
	var store *ssa.Value
	for _, v := range f.f.Blocks[0].Values {
		if v.Op == ssaop.OpStore {
			if store != nil {
				t.Fatalf("expected 1 store (%v), got a second one %v", store, v)
			}
			store = v
		}
	}
	if store == nil {
		t.Fatalf("expected 1 store, got none")
	}

	consT := store.Args[1]
	for ; consT.Op == ssaop.OpCopy; consT = consT.Args[0] {
	}

	if consT.Op != ssaop.OpConst64 || consT.AuxInt != 1 {
		t.Fatalf("expected store to have constant 1, got %v", consT.LongString())
	}
}
