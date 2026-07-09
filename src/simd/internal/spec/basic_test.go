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
