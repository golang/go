// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package spec

import (
	"fmt"
	"slices"
	"testing"
)

func vecOf[E EltOrMask, W Width](xs ...E) Vec[E, W] {
	l := lanes[E, W]()
	if len(xs) != l {
		panic(fmt.Sprintf("got %d elements, want %d", len(xs), l))
	}
	return xs
}

func TestPreserveTNxL(t *testing.T) {
	x := vecOf[int32, Width128](1, 2, 3, 4)
	y := vecOf[int32, Width128](2, 3, 4, 5)
	want := vecOf[int32, Width128](3, 5, 7, 9)
	z := Add(x, y)
	if !slices.Equal(z, want) {
		t.Fatalf("got %v, want %v", z, want)
	}
}

func TestSub(t *testing.T) {
	x := vecOf[int32, Width128](10, 20, 30, 40)
	y := vecOf[int32, Width128](1, 2, 3, 4)
	want := vecOf[int32, Width128](9, 18, 27, 36)
	z := Sub(x, y)
	if !slices.Equal(z, want) {
		t.Fatalf("got %v, want %v", z, want)
	}
}

func TestAddSubSaturated(t *testing.T) {
	x := vecOf[int8, Width128](120, -120, 0, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
	y := vecOf[int8, Width128](10, -10, 0, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
	wantAdd := vecOf[int8, Width128](127, -128, 0, 20, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24)
	if got := AddSaturated(x, y); !slices.Equal(got, wantAdd) {
		t.Fatalf("AddSaturated: got %v, want %v", got, wantAdd)
	}
	wantSub := vecOf[int8, Width128](110, -110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
	if got := SubSaturated(x, y); !slices.Equal(got, wantSub) {
		t.Fatalf("SubSaturated: got %v, want %v", got, wantSub)
	}
}

func TestPreserveL(t *testing.T) {
	// This operation changes T and N
	x := vecOf[int64, Width256](1, 2, 3, 4)
	want := vecOf[float32, Width128](1, 2, 3, 4)
	z := ConvertToZ[int64, Width256, float32, Width128](x)
	if !slices.Equal(z, want) {
		t.Fatalf("got %v, want %v", z, want)
	}
}

func TestPreserveNxL(t *testing.T) {
	// This operation changes T
	x := vecOf[int32, Width128](1, 2, 3, 4)
	want := vecOf[float32, Width128](1, 2, 3, 4)
	z := ConvertToZ[int32, Width128, float32, Width128](x)
	if !slices.Equal(z, want) {
		t.Fatalf("got %v, want %v", z, want)
	}
}

func TestWidthRounding(t *testing.T) {
	// The "natural" result of this is only 64 bits, so it gets rounded up to
	// 128 bits.
	x := vecOf[int64, Width128](1, 2)
	want := vecOf[float32, Width128](1, 2, 0, 0)
	z := ConvertToZ[int64, Width128, float32, Width128](x)
	if !slices.Equal(z, want) {
		t.Fatalf("got %v, want %v", z, want)
	}
}

func TestPreserveW(t *testing.T) {
	x := vecOf[int32, Width128](1, 2, 3, 4)
	y := vecOf[int32, Width128](2, 3, 4, 5)
	want := vecOf[int64, Width128](1*2+2*3, 3*4+4*5)
	z := DotProductPairs[int32, Width128, int64](x, y)
	if !slices.Equal(z, want) {
		t.Fatalf("got %v, want %v", z, want)
	}
}

func TestConcatPairs(t *testing.T) {
	x := vecOf[int32, Width128](1, 2, 3, 4)
	y := vecOf[int32, Width128](5, 6, 7, 8)
	wantAdd := vecOf[int32, Width128](1+2, 3+4, 5+6, 7+8)
	if got := ConcatAddPairs(x, y); !slices.Equal(got, wantAdd) {
		t.Fatalf("ConcatAddPairs: got %v, want %v", got, wantAdd)
	}
	wantSub := vecOf[int32, Width128](1-2, 3-4, 5-6, 7-8)
	if got := ConcatSubPairs(x, y); !slices.Equal(got, wantSub) {
		t.Fatalf("ConcatSubPairs: got %v, want %v", got, wantSub)
	}

	xSat := vecOf[int16, Width128](30000, 10000, -30000, -10000, 10, 20, 30, 40)
	ySat := vecOf[int16, Width128](100, 200, 300, 400, 500, 600, 700, 800)
	wantAddSat := vecOf[int16, Width128](32767, -32768, 30, 70, 300, 700, 1100, 1500)
	if got := ConcatAddPairsSaturated(xSat, ySat); !slices.Equal(got, wantAddSat) {
		t.Fatalf("ConcatAddPairsSaturated: got %v, want %v", got, wantAddSat)
	}
	wantSubSat := vecOf[int16, Width128](20000, -20000, -10, -10, -100, -100, -100, -100)
	if got := ConcatSubPairsSaturated(xSat, ySat); !slices.Equal(got, wantSubSat) {
		t.Fatalf("ConcatSubPairsSaturated: got %v, want %v", got, wantSubSat)
	}
}

func TestConcatPairsGrouped(t *testing.T) {
	x := vecOf[int32, Width256](1, 2, 3, 4, 10, 20, 30, 40)
	y := vecOf[int32, Width256](5, 6, 7, 8, 50, 60, 70, 80)
	wantAdd := vecOf[int32, Width256](1+2, 3+4, 5+6, 7+8, 10+20, 30+40, 50+60, 70+80)
	if got := ConcatAddPairsGrouped(x, y); !slices.Equal(got, wantAdd) {
		t.Fatalf("ConcatAddPairsGrouped: got %v, want %v", got, wantAdd)
	}
	wantSub := vecOf[int32, Width256](1-2, 3-4, 5-6, 7-8, 10-20, 30-40, 50-60, 70-80)
	if got := ConcatSubPairsGrouped(x, y); !slices.Equal(got, wantSub) {
		t.Fatalf("ConcatSubPairsGrouped: got %v, want %v", got, wantSub)
	}

	xSat := vecOf[int16, Width256](
		30000, 10000, -30000, -10000, 10, 20, 30, 40,
		1, 2, 3, 4, 5, 6, 7, 8,
	)
	ySat := vecOf[int16, Width256](
		100, 200, 300, 400, 500, 600, 700, 800,
		10, 20, 30, 40, 50, 60, 70, 80,
	)
	wantAddSat := vecOf[int16, Width256](
		32767, -32768, 30, 70, 300, 700, 1100, 1500,
		3, 7, 11, 15, 30, 70, 110, 150,
	)
	if got := ConcatAddPairsSaturatedGrouped(xSat, ySat); !slices.Equal(got, wantAddSat) {
		t.Fatalf("ConcatAddPairsSaturatedGrouped: got %v, want %v", got, wantAddSat)
	}
	wantSubSat := vecOf[int16, Width256](
		20000, -20000, -10, -10, -100, -100, -100, -100,
		-1, -1, -1, -1, -10, -10, -10, -10,
	)
	if got := ConcatSubPairsSaturatedGrouped(xSat, ySat); !slices.Equal(got, wantSubSat) {
		t.Fatalf("ConcatSubPairsSaturatedGrouped: got %v, want %v", got, wantSubSat)
	}
}

func TestShifts(t *testing.T) {
	// ShiftAllRight
	xSigned := vecOf[int16, Width128](-4, 8, -16, 32, 0, -1, 100, -100)
	wantSigned := vecOf[int16, Width128](-2, 4, -8, 16, 0, -1, 50, -50)
	if got := ShiftAllRight(xSigned, 1); !slices.Equal(got, wantSigned) {
		t.Fatalf("ShiftAllRight signed: got %v, want %v", got, wantSigned)
	}
	wantOOB := vecOf[int16, Width128](-1, 0, -1, 0, 0, -1, 0, -1)
	if got := ShiftAllRight(xSigned, 20); !slices.Equal(got, wantOOB) {
		t.Fatalf("ShiftAllRight signed OOB: got %v, want %v", got, wantOOB)
	}

	xUnsigned := vecOf[uint16, Width128](4, 8, 16, 32, 0, 1, 100, 200)
	wantUnsigned := vecOf[uint16, Width128](2, 4, 8, 16, 0, 0, 50, 100)
	if got := ShiftAllRight(xUnsigned, 1); !slices.Equal(got, wantUnsigned) {
		t.Fatalf("ShiftAllRight unsigned: got %v, want %v", got, wantUnsigned)
	}
	wantUnsignedOOB := vecOf[uint16, Width128](0, 0, 0, 0, 0, 0, 0, 0)
	if got := ShiftAllRight(xUnsigned, 20); !slices.Equal(got, wantUnsignedOOB) {
		t.Fatalf("ShiftAllRight unsigned OOB: got %v, want %v", got, wantUnsignedOOB)
	}

	// ShiftRight
	shiftAmounts := vecOf[uint16, Width128](1, 2, 3, 4, 0, 1, 2, 20)
	wantRightSigned := vecOf[int16, Width128](-4>>1, 8>>2, -16>>3, 32>>4, 0, -1>>1, 100>>2, -1)
	if got := ShiftRight(xSigned, shiftAmounts); !slices.Equal(got, wantRightSigned) {
		t.Fatalf("ShiftRight signed: got %v, want %v", got, wantRightSigned)
	}

	// ScaleSaturated
	// positive shifts left (saturates), negative shifts right
	x8Signed := vecOf[int8, Width128](10, -10, 100, -100, 50, 60, -60, 0, 1, 2, 3, 4, 5, 6, 7, 8)
	y8Signed := vecOf[int8, Width128](1, -1, 2, -2, 0, 1, -1, 0, 1, 1, 1, 1, 1, 1, 1, 1)
	want8Signed := vecOf[int8, Width128](20, -5, 127, -25, 50, 120, -30, 0, 2, 4, 6, 8, 10, 12, 14, 16)
	if got := ScaleSaturated(x8Signed, y8Signed); !slices.Equal(got, want8Signed) {
		t.Fatalf("ScaleSaturated signed: got %v, want %v", got, want8Signed)
	}

	x8Unsigned := vecOf[uint8, Width128](10, 20, 200, 100, 50, 60, 250, 0, 1, 2, 3, 4, 5, 6, 7, 8)
	want8Unsigned := vecOf[uint8, Width128](20, 10, 255, 25, 50, 120, 125, 0, 2, 4, 6, 8, 10, 12, 14, 16)
	if got := ScaleSaturated(x8Unsigned, y8Signed); !slices.Equal(got, want8Unsigned) {
		t.Fatalf("ScaleSaturated unsigned: got %v, want %v", got, want8Unsigned)
	}

	// Wide range (exponents beyond int8)
	x16Signed := vecOf[int16, Width128](10, -10, 10, -10, 10, -10, 0, 1)
	y16Signed := vecOf[int16, Width128](200, 200, -200, -200, 1, -1, 0, 10)
	want16Signed := vecOf[int16, Width128](32767, -32768, 0, -1, 20, -5, 0, 1024)
	if got := ScaleSaturated(x16Signed, y16Signed); !slices.Equal(got, want16Signed) {
		t.Fatalf("ScaleSaturated int16 wide range: got %v, want %v", got, want16Signed)
	}
}
