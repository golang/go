// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"math"
)

// logosSlotTag is the top bit that marks a write barrier buffer entry as the
// address of a stored-to slot; it must equal runtime.logosGenSlotTag.
const logosSlotTag = math.MinInt64

// logosStoresSlot reports whether the pointer store of val needs a card
// entry under GOEXPERIMENT=logosgen: storing nil creates no pointer.
func logosStoresSlot(val *Value) bool {
	return val.Op != OpConstNil
}

// logosSlotEntry returns the write barrier buffer entry of a pointer store
// to ptr: the slot address with logosSlotTag set. b is the write barrier
// block and mem the memory state in it.
func logosSlotEntry(b *Block, pos src.XPos, ptr, mem *Value) *Value {
	slot := b.NewValue2(pos, OpConvert, types.Types[types.TUINTPTR], ptr, mem)
	tag := b.Func.ConstInt64(types.Types[types.TUINTPTR], logosSlotTag)
	return b.NewValue2(pos, OpOr64, types.Types[types.TUINTPTR], slot, tag)
}
