// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd_test

import (
	"fmt"
	"simd"
	"simd/archsimd"
	"testing"
)

func doubleViaArchAMD64(x simd.Int32s) simd.Int32s {
	switch a := x.ToArch().(type) {
	case archsimd.Int32x4:
		return simd.Int32sFromArch(a.Add(a))
	case archsimd.Int32x8:
		return simd.Int32sFromArch(a.Add(a))
	case archsimd.Int32x16:
		return simd.Int32sFromArch(a.Add(a))
	default:
		panic("unexpected ToArch type")
	}
}

func TestToFromArchAMD64(t *testing.T) {
	if simd.Emulated() {
		t.Skip("Emulated() is true")
	}
	s := make([]int32, 16)
	for i := range s {
		s[i] = int32(i + 1)
	}
	x := simd.LoadInt32s(s)
	got := make([]int32, x.Len())
	doubleViaArchAMD64(x).Store(got)
	for i := range got {
		if got[i] != 2*s[i] {
			t.Fatalf("lane %d = %d, want %d", i, got[i], 2*s[i])
		}
	}

	var sink simd.Int32s
	allocs := testing.AllocsPerRun(100, func() {
		sink = doubleViaArchAMD64(x)
	})
	_ = sink
	if allocs != 0 {
		t.Errorf("doubleViaArchAMD64 allocated %v times per run, want 0", allocs)
	}
}

func fromArch128(v archsimd.Int32x4) simd.Int32s { return simd.Int32sFromArch(v) }
func fromArch256(v archsimd.Int32x8) simd.Int32s { return simd.Int32sFromArch(v) }

func TestFromArchWrongWidthPanic(t *testing.T) {
	if simd.Emulated() {
		t.Skip("Emulated() is true; covered by TestFromArchEmulatedPanic")
	}
	bits := simd.VectorBits()
	if bits == 128 {
		want := fmt.Sprintf("simd.Int32sFromArch: cannot convert from archsimd.Int32x8 when simd.VectorBits() is %d", bits)
		expectFromArchPanic(t, func() { _ = fromArch256(archsimd.Int32x8{}) }, want)
	} else {
		want := fmt.Sprintf("simd.Int32sFromArch: cannot convert from archsimd.Int32x4 when simd.VectorBits() is %d", bits)
		expectFromArchPanic(t, func() { _ = fromArch128(archsimd.Int32x4{}) }, want)
	}
}
