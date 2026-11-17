// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"fmt"
	"math/bits"
	"testing"
)

var logosKnownBitsOps = [...]struct {
	width                 uint
	kind                  types.Kind
	add, sub, neg, rotate Op
}{
	{8, types.TUINT8, OpAdd8, OpSub8, OpNeg8, OpRotateLeft8},
	{16, types.TUINT16, OpAdd16, OpSub16, OpNeg16, OpRotateLeft16},
	{32, types.TUINT32, OpAdd32, OpSub32, OpNeg32, OpRotateLeft32},
	{64, types.TUINT64, OpAdd64, OpSub64, OpNeg64, OpRotateLeft64},
}

func enableLogosKnownBits(t *testing.T) {
	t.Helper()
	old := base.Debug.LogosOpt
	base.Debug.LogosOpt = 1
	t.Cleanup(func() { base.Debug.LogosOpt = old })
}

// logosKnownBitsFact uses the same width normalization as the public transfer
// boundary: both the value and its known mask are sign-extended to int64.
func logosKnownBitsFact(value, known uint64, width uint) knownBitsEntry {
	shift := 64 - width
	return knownBitsEntry{
		value: int64(value<<shift) >> shift,
		known: int64(known<<shift) >> shift,
	}
}

// The operands are seeded as cached facts, so these tests exercise fold's
// transfer boundary without depending on another operation to produce facts.
func logosKnownBitsEvaluator(t *testing.T, op Op, valueType, countType *types.Type) func(knownBitsEntry, knownBitsEntry) knownBitsEntry {
	t.Helper()
	c := testConfig(t)
	args := []string{"x", "y"}
	if opcodeTable[op].argLen == 1 {
		args = args[:1]
	}
	fun := c.Fun("entry", Bloc("entry",
		Valu("mem", OpInitMem, types.TypeMem, 0, nil),
		Valu("x", OpArg, valueType, 0, nil),
		Valu("y", OpArg, countType, 0, nil),
		Valu("out", op, valueType, 0, nil, args...),
		Ret("mem")))
	kb := knownBitsState{
		entries:         make([]knownBitsEntry, fun.f.NumValues()),
		seenValues:      newBitset(fun.f.NumValues()),
		reachableBlocks: newBitset(fun.f.NumBlocks()),
	}
	return func(x, y knownBitsEntry) knownBitsEntry {
		clear(kb.entries)
		clear(kb.seenValues)
		xid, yid := fun.values["x"].ID, fun.values["y"].ID
		kb.entries[xid], kb.entries[yid] = x, y
		kb.seenValues.Set(uint32(xid))
		kb.seenValues.Set(uint32(yid))
		value, known := kb.fold(fun.values["out"])
		return knownBitsEntry{value: value, known: known}
	}
}

type logosKnownBitsDomainValue struct {
	fact     knownBitsEntry
	concrete []uint64
}

// Enumerate every three-state assignment in n bits. All other bits are zero.
// Concrete representatives are found by rejection, independently of the
// compiler's allPossibleValues iterator and the transfer implementation.
func logosKnownBitsDomain(n, offset, width uint) []logosKnownBitsDomainValue {
	mask := uint64(1)<<n - 1
	var domain []logosKnownBitsDomainValue
	for known := uint64(0); known <= mask; known++ {
		for value := uint64(0); value <= mask; value++ {
			if value&^known != 0 {
				continue
			}
			d := logosKnownBitsDomainValue{
				fact: logosKnownBitsFact(value<<offset, known<<offset|^(mask<<offset), width),
			}
			for x := uint64(0); x <= mask; x++ {
				if x&known == value {
					d.concrete = append(d.concrete, x<<offset)
				}
			}
			domain = append(domain, d)
		}
	}
	return domain
}

// Intersect concrete results to obtain the best representable abstract result.
func logosKnownBitsOracle(width uint, xs, ys []uint64, op func(uint64, uint64) uint64) knownBitsEntry {
	first := op(xs[0], ys[0])
	var different uint64
	for _, x := range xs {
		for _, y := range ys {
			different |= first ^ op(x, y)
		}
	}
	return logosKnownBitsFact(first&^different, ^different, width)
}

func TestLogosKnownBitsArithmeticDomain(t *testing.T) {
	enableLogosKnownBits(t)
	for _, ops := range logosKnownBitsOps {
		for _, offset := range []uint{0, ops.width - 4} {
			domain := logosKnownBitsDomain(4, offset, ops.width)
			for _, tc := range []struct {
				op    Op
				apply func(uint64, uint64) uint64
			}{
				{ops.add, func(x, y uint64) uint64 { return x + y }},
				{ops.sub, func(x, y uint64) uint64 { return x - y }},
				{ops.neg, func(x, _ uint64) uint64 { return -x }},
			} {
				t.Run(fmt.Sprintf("%s/offset%d", tc.op, offset), func(t *testing.T) {
					typ := types.Types[ops.kind]
					eval := logosKnownBitsEvaluator(t, tc.op, typ, typ)
					rhs := domain
					if tc.op == ops.neg {
						rhs = domain[:1]
					}
					for _, x := range domain {
						for _, y := range rhs {
							want := logosKnownBitsOracle(ops.width, x.concrete, y.concrete, tc.apply)
							if got := eval(x.fact, y.fact); got != want {
								t.Fatalf("%s(%v, %v) = %v, want %v", tc.op, x.fact, y.fact, got, want)
							}
						}
					}
				})
			}
		}
	}
}

func TestLogosKnownBitsArithmeticExtremes(t *testing.T) {
	enableLogosKnownBits(t)
	for _, ops := range logosKnownBitsOps {
		mask := ^uint64(0) >> (64 - ops.width)
		sign := uint64(1) << (ops.width - 1)
		values := []uint64{0, 1, mask, sign, sign - 1, sign + 1, 0xaaaaaaaaaaaaaaaa & mask, 0x5555555555555555 & mask}
		var partial []logosKnownBitsDomainValue
		for _, unknown := range []uint64{1, sign, sign | 1, 1<<(ops.width/2) | 1, sign | 1<<(ops.width/2) | 1} {
			for _, base := range []uint64{0, mask, 0xaaaaaaaaaaaaaaaa & mask} {
				base &^= unknown
				concrete := []uint64{base}
				for bit := uint(0); bit < ops.width; bit++ {
					if unknown&(uint64(1)<<bit) == 0 {
						continue
					}
					for _, value := range concrete {
						concrete = append(concrete, value|uint64(1)<<bit)
					}
				}
				partial = append(partial, logosKnownBitsDomainValue{
					fact:     logosKnownBitsFact(base, ^unknown, ops.width),
					concrete: concrete,
				})
			}
		}
		for _, tc := range []struct {
			op    Op
			apply func(uint64, uint64) uint64
		}{
			{ops.add, func(x, y uint64) uint64 { return x + y }},
			{ops.sub, func(x, y uint64) uint64 { return x - y }},
			{ops.neg, func(x, _ uint64) uint64 { return -x }},
		} {
			t.Run(tc.op.String(), func(t *testing.T) {
				typ := types.Types[ops.kind]
				eval := logosKnownBitsEvaluator(t, tc.op, typ, typ)
				for _, x := range values {
					for _, y := range values {
						want := logosKnownBitsFact(tc.apply(x, y), ^uint64(0), ops.width)
						if got := eval(logosKnownBitsFact(x, mask, ops.width), logosKnownBitsFact(y, mask, ops.width)); got != want {
							t.Fatalf("%s(%#x, %#x) = %v, want %v", tc.op, x, y, got, want)
						}
					}
				}
				for _, x := range partial {
					for _, y := range partial {
						want := logosKnownBitsOracle(ops.width, x.concrete, y.concrete, tc.apply)
						if got := eval(x.fact, y.fact); got != want {
							t.Fatalf("%s(%v, %v) = %v, want %v", tc.op, x.fact, y.fact, got, want)
						}
					}
				}
				if got := eval(knownBitsEntry{}, knownBitsEntry{}); got != (knownBitsEntry{}) {
					t.Fatalf("%s with unknown operands invented facts: %v", tc.op, got)
				}
			})
		}
	}
}

func TestLogosKnownBitsRotateDomain(t *testing.T) {
	enableLogosKnownBits(t)
	for _, ops := range logosKnownBitsOps {
		t.Run(ops.rotate.String(), func(t *testing.T) {
			eval := logosKnownBitsEvaluator(t, ops.rotate, types.Types[ops.kind], types.Types[types.TINT64])
			data := logosKnownBitsDomain(4, ops.width-4, ops.width)
			counts := logosKnownBitsDomain(uint(bits.Len(ops.width)-1), 0, 64)
			rotate := func(x, count uint64) uint64 {
				count &= uint64(ops.width - 1)
				mask := ^uint64(0) >> (64 - ops.width)
				return (x<<count | x>>(uint64(ops.width)-count)) & mask
			}
			for _, x := range data {
				for _, y := range counts {
					// High count bits are unknown, including the sign. They do
					// not affect the residue or any concrete rotate result.
					y.fact.known &= int64(ops.width - 1)
					want := logosKnownBitsOracle(ops.width, x.concrete, y.concrete, rotate)
					if got := eval(x.fact, y.fact); got != want {
						t.Fatalf("rotate(%v, %v) = %v, want %v", x.fact, y.fact, got, want)
					}
				}
			}
			if got := eval(knownBitsEntry{}, knownBitsEntry{}); got != (knownBitsEntry{}) {
				t.Fatalf("rotate with unknown operands invented facts: %v", got)
			}
		})
	}
}

func TestLogosKnownBitsRotateCountWidths(t *testing.T) {
	enableLogosKnownBits(t)
	for _, ops := range logosKnownBitsOps {
		for _, kind := range []types.Kind{types.TINT8, types.TUINT16, types.TINT32, types.TUINT64} {
			t.Run(fmt.Sprintf("%s/%s", ops.rotate, types.Types[kind]), func(t *testing.T) {
				x := uint64(0x8123456789abcdef) >> (64 - ops.width)
				countType := types.Types[kind]
				eval := logosKnownBitsEvaluator(t, ops.rotate, types.Types[ops.kind], countType)
				for _, count := range []int64{-1, 0, 1, int64(ops.width), int64(ops.width) + 1, 1 << 62} {
					y := logosKnownBitsFact(uint64(count), ^uint64(0), uint(countType.Size()*8))
					shift := uint64(y.value) & uint64(ops.width-1)
					want := logosKnownBitsFact(x<<shift|x>>(uint64(ops.width)-shift), ^uint64(0), ops.width)
					if got := eval(logosKnownBitsFact(x, ^uint64(0), ops.width), y); got != want {
						t.Fatalf("rotate(%#x, %d) = %v, want %v", x, y.value, got, want)
					}
				}
			})
		}
	}
}

func TestLogosKnownBitsDisabled(t *testing.T) {
	old := base.Debug.LogosOpt
	base.Debug.LogosOpt = 0
	t.Cleanup(func() { base.Debug.LogosOpt = old })
	for _, ops := range logosKnownBitsOps {
		for _, op := range []Op{ops.add, ops.sub, ops.neg, ops.rotate} {
			typ := types.Types[ops.kind]
			eval := logosKnownBitsEvaluator(t, op, typ, typ)
			if got := eval(logosKnownBitsFact(3, ^uint64(0), ops.width), logosKnownBitsFact(1, ^uint64(0), ops.width)); got != (knownBitsEntry{}) {
				t.Fatalf("disabled %s transfer changed upstream facts: %v", op, got)
			}
		}
	}
}

func TestLogosKnownBitsFoldMaskedArithmetic(t *testing.T) {
	enableLogosKnownBits(t)
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("x", OpArg, c.config.Types.UInt64, 0, nil),
			Valu("high", OpConst64, c.config.Types.UInt64, -16, nil),
			Valu("low", OpConst64, c.config.Types.UInt64, 15, nil),
			Valu("three", OpConst64, c.config.Types.UInt64, 3, nil),
			Valu("aligned", OpAnd64, c.config.Types.UInt64, 0, nil, "x", "high"),
			Valu("sum", OpAdd64, c.config.Types.UInt64, 0, nil, "aligned", "three"),
			Valu("masked", OpAnd64, c.config.Types.UInt64, 0, nil, "sum", "low"),
			Valu("equal", OpEq64, c.config.Types.Bool, 0, nil, "masked", "three"),
			If("equal", "yes", "no")),
		Bloc("yes", Ret("mem")),
		Bloc("no", Ret("mem")))
	knownBits(fun.f)
	copyelim(fun.f)
	got := fun.blocks["entry"].Controls[0]
	if got.Op != OpConstBool || got.AuxInt != 1 {
		t.Fatalf("masked arithmetic condition = %v, want constant true", got)
	}
}

func TestLogosKnownBitsLoopConservative(t *testing.T) {
	enableLogosKnownBits(t)
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("zero", OpConst8, c.config.Types.UInt8, 0, nil),
			Valu("one", OpConst8, c.config.Types.UInt8, 1, nil),
			Valu("condition", OpArg, c.config.Types.Bool, 0, nil),
			Goto("loop")),
		Bloc("loop",
			Valu("phi", OpPhi, c.config.Types.UInt8, 0, nil, "zero", "next"),
			Valu("next", OpAdd8, c.config.Types.UInt8, 0, nil, "phi", "one"),
			If("condition", "loop", "exit")),
		Bloc("exit", Ret("mem")))
	kb := knownBitsState{
		entries:         make([]knownBitsEntry, fun.f.NumValues()),
		seenValues:      newBitset(fun.f.NumValues()),
		reachableBlocks: newBitset(fun.f.NumBlocks()),
	}
	for _, b := range fun.f.Blocks {
		kb.reachableBlocks.Set(uint32(b.ID))
	}
	value, known := kb.fold(fun.values["phi"])
	if value != 0 || known != 0 {
		t.Fatalf("wrapped induction variable has invented facts: value=%x known=%x", value, known)
	}
}
