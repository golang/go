// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package frob_test

import (
	"math"
	"reflect"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/frob"
)

func TestBasics(t *testing.T) {
	type Basics struct {
		A []*string
		B [2]int
		C *Basics
		D map[string]int
	}
	codec := frob.CodecFor117(new(Basics))

	s1, s2 := "hello", "world"
	x := Basics{
		A: []*string{&s1, nil, &s2},
		B: [...]int{1, 2},
		C: &Basics{
			B: [...]int{3, 4},
			D: map[string]int{"one": 1},
		},
	}
	var y Basics
	codec.Decode(codec.Encode(x), &y)
	if !reflect.DeepEqual(x, y) {
		t.Fatalf("bad roundtrip: got %#v, want %#v", y, x)
	}
}

func TestInts(t *testing.T) {
	type Ints struct {
		U    uint
		U8   uint8
		U16  uint16
		U32  uint32
		U64  uint64
		UP   uintptr
		I    int
		I8   int8
		I16  int16
		I32  int32
		I64  int64
		F32  float32
		F64  float64
		C64  complex64
		C128 complex128
	}
	codec := frob.CodecFor117(new(Ints))

	// maxima
	max1 := Ints{
		U:    math.MaxUint,
		U8:   math.MaxUint8,
		U16:  math.MaxUint16,
		U32:  math.MaxUint32,
		U64:  math.MaxUint64,
		UP:   math.MaxUint,
		I:    math.MaxInt,
		I8:   math.MaxInt8,
		I16:  math.MaxInt16,
		I32:  math.MaxInt32,
		I64:  math.MaxInt64,
		F32:  math.MaxFloat32,
		F64:  math.MaxFloat64,
		C64:  complex(math.MaxFloat32, math.MaxFloat32),
		C128: complex(math.MaxFloat64, math.MaxFloat64),
	}
	var max2 Ints
	codec.Decode(codec.Encode(max1), &max2)
	if !reflect.DeepEqual(max1, max2) {
		t.Fatalf("max: bad roundtrip: got %#v, want %#v", max2, max1)
	}

	// minima
	min1 := Ints{
		I:    math.MinInt,
		I8:   math.MinInt8,
		I16:  math.MinInt16,
		I32:  math.MinInt32,
		I64:  math.MinInt64,
		F32:  -math.MaxFloat32,
		F64:  -math.MaxFloat32,
		C64:  complex(-math.MaxFloat32, -math.MaxFloat32),
		C128: complex(-math.MaxFloat64, -math.MaxFloat64),
	}
	var min2 Ints
	codec.Decode(codec.Encode(min1), &min2)
	if !reflect.DeepEqual(min1, min2) {
		t.Fatalf("min: bad roundtrip: got %#v, want %#v", min2, min1)
	}

	// negatives (other than MinInt), to exercise conversions
	neg1 := Ints{
		I:   -1,
		I8:  -1,
		I16: -1,
		I32: -1,
		I64: -1,
	}
	var neg2 Ints
	codec.Decode(codec.Encode(neg1), &neg2)
	if !reflect.DeepEqual(neg1, neg2) {
		t.Fatalf("neg: bad roundtrip: got %#v, want %#v", neg2, neg1)
	}
}
