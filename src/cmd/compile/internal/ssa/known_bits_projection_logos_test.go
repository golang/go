// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"fmt"
	"testing"
)

var logosProjectionOps = [...]struct {
	width                     uint
	kind                      types.Kind
	constant, and, or, rotate Op
	right, signed             [4]Op
}{
	{8, types.TUINT8, OpConst8, OpAnd8, OpOr8, OpRotateLeft8,
		[4]Op{OpRsh8Ux8, OpRsh8Ux16, OpRsh8Ux32, OpRsh8Ux64},
		[4]Op{OpRsh8x8, OpRsh8x16, OpRsh8x32, OpRsh8x64}},
	{16, types.TUINT16, OpConst16, OpAnd16, OpOr16, OpRotateLeft16,
		[4]Op{OpRsh16Ux8, OpRsh16Ux16, OpRsh16Ux32, OpRsh16Ux64},
		[4]Op{OpRsh16x8, OpRsh16x16, OpRsh16x32, OpRsh16x64}},
	{32, types.TUINT32, OpConst32, OpAnd32, OpOr32, OpRotateLeft32,
		[4]Op{OpRsh32Ux8, OpRsh32Ux16, OpRsh32Ux32, OpRsh32Ux64},
		[4]Op{OpRsh32x8, OpRsh32x16, OpRsh32x32, OpRsh32x64}},
	{64, types.TUINT64, OpConst64, OpAnd64, OpOr64, OpRotateLeft64,
		[4]Op{OpRsh64Ux8, OpRsh64Ux16, OpRsh64Ux32, OpRsh64Ux64},
		[4]Op{OpRsh64x8, OpRsh64x16, OpRsh64x32, OpRsh64x64}},
}

func logosProjectionFunc(t *testing.T, widthIndex, rotateCountIndex, shiftCountIndex int, count int64, keep, fixed uint64) fun {
	t.Helper()
	c := testConfig(t)
	ops := logosProjectionOps[widthIndex]
	rc, sc := logosProjectionOps[rotateCountIndex], logosProjectionOps[shiftCountIndex]
	typ := types.Types[ops.kind]
	return c.Fun("entry", Bloc("entry",
		Valu("mem", OpInitMem, types.TypeMem, 0, nil),
		Valu("ptr", OpArg, types.NewPtr(typ), 0, nil),
		Valu("raw", OpArg, typ, 0, nil),
		Valu("mask", ops.constant, typ, int64(keep), nil),
		Valu("fixed", ops.constant, typ, int64(fixed), nil),
		Valu("masked", ops.and, typ, 0, nil, "raw", "mask"),
		Valu("input", ops.or, typ, 0, nil, "masked", "fixed"),
		Valu("rotateCount", rc.constant, types.Types[rc.kind], count, nil),
		Valu("shiftCount", sc.constant, types.Types[sc.kind], count, nil),
		Valu("rotated", ops.rotate, typ, 0, nil, "input", "rotateCount"),
		Valu("out", ops.right[shiftCountIndex], typ, 0, nil, "rotated", "shiftCount"),
		Valu("store", OpStore, types.TypeMem, 0, typ, "ptr", "out", "mem"),
		Ret("store")))
}

// Normalize after case-specific edits, then validate both the fixture and the
// transformed graph. Scalar constant AuxInt fields are signed at their width.
func runLogosProjection(f *Func) {
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			switch v.Op {
			case OpConst8:
				v.AuxInt = int64(int8(v.AuxInt))
			case OpConst16:
				v.AuxInt = int64(int16(v.AuxInt))
			case OpConst32:
				v.AuxInt = int64(int32(v.AuxInt))
			}
		}
	}
	CheckFunc(f)
	knownBits(f)
	CheckFunc(f)
}

func TestLogosKnownBitsProjectionWidths(t *testing.T) {
	enableLogosKnownBits(t)
	for wi, ops := range logosProjectionOps {
		mask := ^uint64(0) >> (64 - ops.width)
		for ri, rc := range logosProjectionOps {
			for si, sc := range logosProjectionOps {
				for _, count := range []uint{0, 1, ops.width / 2, ops.width - 1} {
					t.Run(fmt.Sprintf("width%d/rotate%d/shift%d/k%d", ops.width, rc.width, sc.width, count), func(t *testing.T) {
						// At least one input bit stays dynamic, including at
						// width-1. This requires an identity, not constant folding.
						fun := logosProjectionFunc(t, wi, ri, si, int64(count), mask>>count, 0)
						n := fun.f.NumValues()
						runLogosProjection(fun.f)
						out := fun.values["out"]
						if out.Op != OpCopy || len(out.Args) != 1 || out.Args[0] != fun.values["input"] {
							t.Fatalf("projection = %s, want copy of input", out.LongString())
						}
						if got := fun.f.NumValues(); got != n {
							t.Fatalf("projection allocated values: %d -> %d", n, got)
						}
					})
				}
			}
		}
	}
}

func TestLogosKnownBitsProjectionGuards(t *testing.T) {
	enableLogosKnownBits(t)
	for wi, ops := range logosProjectionOps {
		mask := ^uint64(0) >> (64 - ops.width)
		count := ops.width / 2
		low := mask >> count
		for si, sc := range logosProjectionOps {
			for _, tc := range []struct {
				name string
				edit func(fun)
			}{
				{"unknown-high", func(f fun) { f.values["mask"].AuxInt = int64(mask) }},
				{"one-unknown-high", func(f fun) { f.values["mask"].AuxInt = int64(low | 1<<(ops.width-count)) }},
				{"one-nonzero-high", func(f fun) { f.values["fixed"].AuxInt = int64(uint64(1) << (ops.width - count)) }},
				{"nonzero-sign", func(f fun) { f.values["fixed"].AuxInt = int64(uint64(1) << (ops.width - 1)) }},
				{"different-counts", func(f fun) { f.values["shiftCount"].AuxInt-- }},
				{"same-residue-different-counts", func(f fun) { f.values["rotateCount"].AuxInt += int64(ops.width) }},
				{"unknown-rotate-count", func(f fun) { f.values["rotateCount"].reset(OpArg) }},
				{"unknown-shift-count", func(f fun) { f.values["shiftCount"].reset(OpArg) }},
				{"nonliteral-rotate-count", func(f fun) {
					f.values["rotateCount"].reset(OpCopy)
					f.values["rotateCount"].AddArg(f.values["shiftCount"])
				}},
				{"nonliteral-shift-count", func(f fun) {
					f.values["shiftCount"].reset(OpCopy)
					f.values["shiftCount"].AddArg(f.values["rotateCount"])
				}},
				{"signed-shift", func(f fun) { f.values["out"].Op = ops.signed[si] }},
				{"no-rotate", func(f fun) { f.values["out"].SetArg(0, f.values["raw"]) }},
			} {
				t.Run(fmt.Sprintf("width%d/count%d/%s", ops.width, sc.width, tc.name), func(t *testing.T) {
					fun := logosProjectionFunc(t, wi, si, si, int64(count), low, 0)
					tc.edit(fun)
					op := fun.values["out"].Op
					runLogosProjection(fun.f)
					if out := fun.values["out"]; out.Op != op {
						t.Fatalf("guarded projection changed %s to %s", op, out.LongString())
					}
				})
			}
		}
	}
}

func TestLogosKnownBitsProjectionOutOfRange(t *testing.T) {
	enableLogosKnownBits(t)
	for wi, ops := range logosProjectionOps {
		mask := ^uint64(0) >> (64 - ops.width)
		for ci, counts := range logosProjectionOps {
			for _, count := range []int64{-1, int64(ops.width), int64(ops.width) + 1} {
				t.Run(fmt.Sprintf("width%d/count%d/k%d", ops.width, counts.width, count), func(t *testing.T) {
					fun := logosProjectionFunc(t, wi, ci, ci, count, mask>>1, 0)
					runLogosProjection(fun.f)
					out := fun.values["out"]
					// The existing constant folder may prove the oversized
					// shift zero; it must never replace it with dynamic input.
					if out.Op != OpCopy || out.Args[0].Op != ops.constant || out.Args[0].AuxInt != 0 {
						t.Fatalf("oversized projection = %s, want zero", out.LongString())
					}
				})
			}
		}
	}
}

func TestLogosKnownBitsProjectionDisabled(t *testing.T) {
	old := base.Debug.LogosOpt
	base.Debug.LogosOpt = 0
	t.Cleanup(func() { base.Debug.LogosOpt = old })
	for wi, ops := range logosProjectionOps {
		mask := ^uint64(0) >> (64 - ops.width)
		for ci := range logosProjectionOps {
			fun := logosProjectionFunc(t, wi, ci, ci, 1, mask>>1, 0)
			runLogosProjection(fun.f)
			if out := fun.values["out"]; out.Op != ops.right[ci] {
				t.Fatalf("disabled projection changed %s", out.LongString())
			}
		}
	}
}

func TestLogosKnownBitsProjectionPackedHeader(t *testing.T) {
	enableLogosKnownBits(t)
	c := testConfig(t)
	typ := c.config.Types.UInt64
	fun := c.Fun("entry", Bloc("entry",
		Valu("mem", OpInitMem, types.TypeMem, 0, nil),
		Valu("ptr", OpArg, types.NewPtr(typ), 0, nil),
		Valu("key", OpArg, c.config.Types.UInt32, 0, nil),
		Valu("sequence", OpArg, c.config.Types.UInt16, 0, nil),
		Valu("count", OpConst64, typ, 16, nil),
		Valu("wideKey", OpZeroExt32to64, typ, 0, nil, "key"),
		Valu("wideSequence", OpZeroExt16to64, typ, 0, nil, "sequence"),
		Valu("upper", OpLsh64x64, typ, 0, nil, "wideKey", "count"),
		Valu("input", OpOr64, typ, 0, nil, "upper", "wideSequence"),
		Valu("rotated", OpRotateLeft64, typ, 0, nil, "input", "count"),
		Valu("out", OpRsh64Ux64, typ, 0, nil, "rotated", "count"),
		Valu("store", OpStore, types.TypeMem, 0, typ, "ptr", "out", "mem"),
		Ret("store")))
	runLogosProjection(fun.f)
	if out := fun.values["out"]; out.Op != OpCopy || out.Args[0] != fun.values["input"] {
		t.Fatalf("packed header projection = %s, want native header", out.LongString())
	}
}

func TestLogosKnownBitsProjectionTruncatedRotate(t *testing.T) {
	enableLogosKnownBits(t)
	c := testConfig(t)
	u32, u64 := c.config.Types.UInt32, c.config.Types.UInt64
	fun := c.Fun("entry", Bloc("entry",
		Valu("mem", OpInitMem, types.TypeMem, 0, nil),
		Valu("ptr", OpArg, types.NewPtr(u32), 0, nil),
		Valu("raw", OpArg, u64, 0, nil),
		Valu("mask", OpConst64, u64, 1<<56-1, nil),
		Valu("count", OpConst8, c.config.Types.UInt8, 8, nil),
		Valu("input", OpAnd64, u64, 0, nil, "raw", "mask"),
		Valu("rotated", OpRotateLeft64, u64, 0, nil, "input", "count"),
		Valu("truncated", OpTrunc64to32, u32, 0, nil, "rotated"),
		Valu("out", OpRsh32Ux8, u32, 0, nil, "truncated", "count"),
		Valu("store", OpStore, types.TypeMem, 0, u32, "ptr", "out", "mem"),
		Ret("store")))
	runLogosProjection(fun.f)
	if out := fun.values["out"]; out.Op != OpRsh32Ux8 {
		t.Fatalf("projection crossed truncation: %s", out.LongString())
	}
}

func TestLogosKnownBitsProjectionCycle(t *testing.T) {
	enableLogosKnownBits(t)
	for _, count := range []int64{0, 1} {
		t.Run(fmt.Sprint(count), func(t *testing.T) {
			c := testConfig(t)
			typ := c.config.Types.UInt64
			fun := c.Fun("entry",
				Bloc("entry",
					Valu("mem", OpInitMem, types.TypeMem, 0, nil),
					Valu("raw", OpArg, typ, 0, nil),
					Valu("count", OpConst64, typ, count, nil),
					Valu("condition", OpArg, c.config.Types.Bool, 0, nil),
					Goto("loop")),
				Bloc("loop",
					Valu("input", OpPhi, typ, 0, nil, "raw", "out"),
					Valu("rotated", OpRotateLeft64, typ, 0, nil, "input", "count"),
					Valu("out", OpRsh64Ux64, typ, 0, nil, "rotated", "count"),
					If("condition", "loop", "exit")),
				Bloc("exit", Ret("mem")))
			runLogosProjection(fun.f)
			out := fun.values["out"]
			if count == 0 {
				if out.Op != OpCopy || out.Args[0] != fun.values["input"] {
					t.Fatalf("zero-count cyclic projection = %s, want copy of input", out.LongString())
				}
			} else if out.Op != OpRsh64Ux64 {
				t.Fatalf("cyclic input acquired unsupported high-zero facts: %s", out.LongString())
			}
		})
	}
}
