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

var logosBitFactsOps = [...]struct {
	width                                 uint
	kind                                  types.Kind
	length, trailing, nonzero, population Op
}{
	{8, types.TUINT8, OpBitLen8, OpCtz8, OpCtz8NonZero, OpPopCount8},
	{16, types.TUINT16, OpBitLen16, OpCtz16, OpCtz16NonZero, OpPopCount16},
	{32, types.TUINT32, OpBitLen32, OpCtz32, OpCtz32NonZero, OpPopCount32},
	{64, types.TUINT64, OpBitLen64, OpCtz64, OpCtz64NonZero, OpPopCount64},
}

// The oracle uses bit-by-bit concrete evaluation, independently of the
// word-level intrinsics and interval-prefix formula used by the transfer.
func logosBitFactsConcrete(op Op, width uint, x uint64) int64 {
	x &= ^uint64(0) >> (64 - width)
	var length, population int64
	trailing := int64(width)
	for bit := uint(0); bit < width; bit++ {
		if x&(uint64(1)<<bit) != 0 {
			length = int64(bit + 1)
			population++
			if trailing == int64(width) {
				trailing = int64(bit)
			}
		}
	}
	switch op {
	case OpBitLen8, OpBitLen16, OpBitLen32, OpBitLen64:
		return length
	case OpCtz8, OpCtz16, OpCtz32, OpCtz64, OpCtz8NonZero, OpCtz16NonZero, OpCtz32NonZero, OpCtz64NonZero:
		return trailing
	case OpPopCount8, OpPopCount16, OpPopCount32, OpPopCount64:
		return population
	default:
		panic("unsupported bit-facts test operation")
	}
}

func logosBitFactsInterval(op Op, width, resultWidth uint, concrete []uint64) knownBitsEntry {
	low := logosBitFactsConcrete(op, width, concrete[0])
	high := low
	for _, x := range concrete[1:] {
		y := logosBitFactsConcrete(op, width, x)
		low = min(low, y)
		high = max(high, y)
	}
	known := int64(-1)
	for y := low; y <= high; y++ {
		known &^= low ^ y
	}
	return logosKnownBitsFact(uint64(low&known), uint64(known), resultWidth)
}

func logosBitFactsEvaluator(t *testing.T, op Op, inputType, resultType *types.Type) func(knownBitsEntry) knownBitsEntry {
	t.Helper()
	c := testConfig(t)
	fun := c.Fun("entry", Bloc("entry",
		Valu("mem", OpInitMem, types.TypeMem, 0, nil),
		Valu("ptr", OpArg, types.NewPtr(resultType), 0, nil),
		Valu("input", OpArg, inputType, 0, nil),
		Valu("out", op, resultType, 0, nil, "input"),
		Valu("store", OpStore, types.TypeMem, 0, resultType, "ptr", "out", "mem"),
		Ret("store")))
	CheckFunc(fun.f)
	kb := knownBitsState{
		entries:         make([]knownBitsEntry, fun.f.NumValues()),
		seenValues:      newBitset(fun.f.NumValues()),
		reachableBlocks: newBitset(fun.f.NumBlocks()),
	}
	return func(input knownBitsEntry) knownBitsEntry {
		clear(kb.entries)
		clear(kb.seenValues)
		id := fun.values["input"].ID
		kb.entries[id] = input
		kb.seenValues.Set(uint32(id))
		value, known := kb.fold(fun.values["out"])
		return knownBitsEntry{value: value, known: known}
	}
}

func TestLogosKnownBitsIntrinsicsDomain(t *testing.T) {
	enableLogosKnownBits(t)
	for _, ops := range logosBitFactsOps {
		domains := [][]logosKnownBitsDomainValue{
			logosKnownBitsDomain(4, 0, ops.width),
			logosKnownBitsDomain(4, ops.width-4, ops.width),
		}
		if ops.width == 8 {
			domains = append(domains, logosKnownBitsDomain(8, 0, 8))
		}
		for _, resultKind := range []types.Kind{types.TINT32, types.TINT64} {
			resultType := types.Types[resultKind]
			resultWidth := uint(resultType.Size() * 8)
			for _, op := range []Op{ops.length, ops.trailing, ops.nonzero, ops.population} {
				t.Run(fmt.Sprintf("%s/result%d", op, resultWidth), func(t *testing.T) {
					eval := logosBitFactsEvaluator(t, op, types.Types[ops.kind], resultType)
					for _, domain := range domains {
						for _, input := range domain {
							if op == ops.nonzero && len(input.concrete) == 1 && input.concrete[0] == 0 {
								continue // a known-zero singleton violates this opcode's precondition
							}
							got := eval(input.fact)
							// Keep zero-inclusive bounds for NonZero when the
							// lattice cannot see its path-dependent precondition.
							want := logosBitFactsInterval(op, ops.width, resultWidth, input.concrete)
							if got != want {
								t.Fatalf("%s(%v) = %v, want interval facts %v", op, input.fact, got, want)
							}
							for _, x := range input.concrete {
								if op == ops.nonzero && x == 0 {
									continue
								}
								y := logosBitFactsConcrete(op, ops.width, x)
								if y&got.known != got.value {
									t.Fatalf("unsound %s facts %v for input %#x, result %d", op, got, x, y)
								}
							}
						}
					}
				})
			}
		}
	}
}

func TestLogosKnownBitsIntrinsicsExtremes(t *testing.T) {
	enableLogosKnownBits(t)
	for _, ops := range logosBitFactsOps {
		mask := ^uint64(0) >> (64 - ops.width)
		values := []uint64{0, 1, 2, mask, mask >> 1, uint64(1) << (ops.width - 1), 0xaaaaaaaaaaaaaaaa & mask, 0x5555555555555555 & mask}
		for _, resultKind := range []types.Kind{types.TINT32, types.TINT64} {
			for _, op := range []Op{ops.length, ops.trailing, ops.nonzero, ops.population} {
				t.Run(fmt.Sprintf("%s/result%d", op, types.Types[resultKind].Size()*8), func(t *testing.T) {
					eval := logosBitFactsEvaluator(t, op, types.Types[ops.kind], types.Types[resultKind])
					for _, x := range values {
						if op == ops.nonzero && x == 0 {
							continue
						}
						want := knownBitsEntry{value: logosBitFactsConcrete(op, ops.width, x), known: -1}
						if got := eval(logosKnownBitsFact(x, mask, ops.width)); got != want {
							t.Fatalf("%s(%#x) = %v, want %v", op, x, got, want)
						}
					}
					unknown := knownBitsEntry{}
					want := logosBitFactsInterval(op, ops.width, uint(types.Types[resultKind].Size()*8), []uint64{0, mask, 1})
					if got := eval(unknown); got != want {
						t.Fatalf("%s without input facts = %v, want bounded result facts %v", op, got, want)
					}
				})
			}
		}
	}
}

func TestLogosKnownBitsIntrinsicsSourceShapes(t *testing.T) {
	enableLogosKnownBits(t)
	c := testConfig(t)
	u64, i64 := c.config.Types.UInt64, c.config.Types.Int64
	fixedOnes := uint64(0xaaaaaaaa00000000)
	fun := c.Fun("entry", Bloc("entry",
		Valu("mem", OpInitMem, types.TypeMem, 0, nil),
		Valu("ptr", OpArg, types.NewPtr(i64), 0, nil),
		Valu("raw", OpArg, u64, 0, nil),
		Valu("key", OpArg, c.config.Types.UInt16, 0, nil),
		Valu("alignedMask", OpConst64, u64, -256, nil),
		Valu("tag", OpConst64, u64, 32, nil),
		Valu("marker", OpConst64, u64, 1<<16, nil),
		Valu("rotateCount", OpConst64, u64, 24, nil),
		Valu("sparseMask", OpConst64, u64, 0x15555555, nil),
		Valu("fixedOnes", OpConst64, u64, int64(fixedOnes), nil),
		Valu("four", OpConst64, u64, 4, nil),
		Valu("aligned", OpAnd64, u64, 0, nil, "raw", "alignedMask"),
		Valu("tagged", OpAdd64, u64, 0, nil, "aligned", "tag"),
		Valu("trailing", OpCtz64, i64, 0, nil, "tagged"),
		Valu("nonzero", OpCtz64NonZero, i64, 0, nil, "tagged"),
		Valu("wideKey", OpZeroExt16to64, u64, 0, nil, "key"),
		Valu("native", OpOr64, u64, 0, nil, "wideKey", "marker"),
		Valu("wire", OpRotateLeft64, u64, 0, nil, "native", "rotateCount"),
		Valu("length", OpBitLen64, i64, 0, nil, "wire"),
		Valu("sparse", OpAnd64, u64, 0, nil, "raw", "sparseMask"),
		Valu("word", OpOr64, u64, 0, nil, "sparse", "fixedOnes"),
		Valu("population", OpPopCount64, i64, 0, nil, "word"),
		Valu("prefix", OpRsh64x64, i64, 0, nil, "population", "four"),
		Valu("storeTrailing", OpStore, types.TypeMem, 0, i64, "ptr", "trailing", "mem"),
		Valu("storeNonzero", OpStore, types.TypeMem, 0, i64, "ptr", "nonzero", "storeTrailing"),
		Valu("storeLength", OpStore, types.TypeMem, 0, i64, "ptr", "length", "storeNonzero"),
		Valu("storePrefix", OpStore, types.TypeMem, 0, i64, "ptr", "prefix", "storeLength"),
		Ret("storePrefix")))
	CheckFunc(fun.f)
	knownBits(fun.f)
	CheckFunc(fun.f)
	for _, tc := range []struct {
		name string
		want int64
	}{{"trailing", 5}, {"nonzero", 5}, {"length", 41}, {"prefix", 1}} {
		out := fun.values[tc.name]
		if out.Op != OpCopy || out.Args[0].Op != OpConst64 || out.Args[0].AuxInt != tc.want {
			t.Errorf("%s = %s, want %d", tc.name, out.LongString(), tc.want)
		}
	}
	if fun.values["population"].Op != OpPopCount64 {
		t.Fatal("a population range was incorrectly replaced with a single count")
	}
}

func TestLogosKnownBitsIntrinsicsDisabled(t *testing.T) {
	old := base.Debug.LogosOpt
	base.Debug.LogosOpt = 0
	t.Cleanup(func() { base.Debug.LogosOpt = old })
	for _, ops := range logosBitFactsOps {
		for _, op := range []Op{ops.length, ops.trailing, ops.nonzero, ops.population} {
			eval := logosBitFactsEvaluator(t, op, types.Types[ops.kind], types.Types[types.TINT64])
			if got := eval(logosKnownBitsFact(1, ^uint64(0), ops.width)); got != (knownBitsEntry{}) {
				t.Fatalf("disabled %s acquired new facts: %v", op, got)
			}
		}
	}
}

func TestLogosKnownBitsIntrinsicsCycle(t *testing.T) {
	enableLogosKnownBits(t)
	for _, op := range []Op{OpBitLen64, OpCtz64, OpPopCount64} {
		t.Run(op.String(), func(t *testing.T) {
			c := testConfig(t)
			u64, i64 := c.config.Types.UInt64, c.config.Types.Int64
			fun := c.Fun("entry",
				Bloc("entry",
					Valu("mem", OpInitMem, types.TypeMem, 0, nil),
					Valu("ptr", OpArg, types.NewPtr(i64), 0, nil),
					Valu("initial", OpArg, u64, 0, nil),
					Valu("one", OpConst64, u64, 1, nil),
					Valu("condition", OpArg, c.config.Types.Bool, 0, nil),
					Goto("loop")),
				Bloc("loop",
					Valu("input", OpPhi, u64, 0, nil, "initial", "next"),
					Valu("next", OpAdd64, u64, 0, nil, "input", "one"),
					Valu("out", op, i64, 0, nil, "input"),
					If("condition", "loop", "exit")),
				Bloc("exit",
					Valu("store", OpStore, types.TypeMem, 0, i64, "ptr", "out", "mem"),
					Ret("store")))
			CheckFunc(fun.f)
			knownBits(fun.f)
			CheckFunc(fun.f)
			if out := fun.values["out"]; out.Op != op {
				t.Fatalf("cyclic input invented a constant: %s", out.LongString())
			}
		})
	}
}
