// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

import (
	"testing"
)

func TestPR3CandidateNonConstNegativeShift(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{
		src:    "package p\nfunc ShiftNonConst(k int) int { return -3 >> k }",
		ptypes: []*Type{BasicType("int")},
		rtypes: []*Type{BasicType("int")},
		name:   "ShiftNonConst",
		call:   `ShiftNonConst(1)`,
	})
	if err != nil {
		t.Fatalf("ShiftNonConst(1): %v", err)
	}
	mustInt(t, v, -2, "-3 >> k (k=1)")
}

func TestPR3CandidateConstNegativeShift(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{
		src:    "package p\nfunc ShiftConstNeg() int { return (-1) >> uint8(1) }",
		rtypes: []*Type{BasicType("int")},
		name:   "ShiftConstNeg",
		call:   `ShiftConstNeg()`,
	})
	if err != nil {
		t.Fatalf("ShiftConstNeg(): %v", err)
	}
	mustInt(t, v, -1, "(-1) >> uint8(1)")
}

func TestPR3CandidateTypedFloatConstEquality(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{
		src:    "package p\nfunc FloatTypedEq() int { if float64(0.1) == 0.1 { return 1 }; return 0 }",
		rtypes: []*Type{BasicType("int")},
		name:   "FloatTypedEq",
		call:   `FloatTypedEq()`,
	})
	if err != nil {
		t.Fatalf("FloatTypedEq(): %v", err)
	}
	mustInt(t, v, 1, "float64(0.1) == 0.1")
}

func TestPR3CandidateExactUntypedDifferingDecimals(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{
		src:    "package p\nfunc DifferingDecimals() int { if 0.1000000000000000000000000001 == 0.1 { return 1 }; return 0 }",
		rtypes: []*Type{BasicType("int")},
		name:   "DifferingDecimals",
		call:   `DifferingDecimals()`,
	})
	if err != nil {
		t.Fatalf("DifferingDecimals(): %v", err)
	}
	mustInt(t, v, 0, "0.100...01 == 0.1")

	vCtrl, _, errCtrl := evalConstFunc(t, cfg, constSpec{
		src:    "package p\nfunc DifferingDecimalsCtrl() int { if 0.1 == 0.1 { return 1 }; return 0 }",
		rtypes: []*Type{BasicType("int")},
		name:   "DifferingDecimalsCtrl",
		call:   `DifferingDecimalsCtrl()`,
	})
	if errCtrl != nil {
		t.Fatalf("DifferingDecimalsCtrl(): %v", errCtrl)
	}
	mustInt(t, vCtrl, 1, "0.1 == 0.1")
}

func TestPR3CandidateRangeHugeUint64BreaksNotZero(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{
		src: `package p
func HugeRangeBreak() int {
	iter := 0
	for range uint64(1) << 63 {
		iter = 1
		break
	}
	return iter
}`,
		rtypes: []*Type{BasicType("int")},
		name:   "HugeRangeBreak",
		call:   `HugeRangeBreak()`,
	})
	if err != nil {
		return
	}
	if v.Int64() == 0 {
		t.Fatalf("HugeRangeBreak: folded zero iterations (success zero), want first iteration executed or admitted error")
	}
	mustInt(t, v, 1, "HugeRangeBreak")
}
