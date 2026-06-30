// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package testdata_test

import (
	"fmt"
	"simd"
	"testing"
)

func fillUint8s(f func(i int) uint8) simd.Uint8s {
	l := simd.Uint8s{}.Len()
	x := make([]uint8, l, l)
	for i := range l {
		x[i] = f(i)
	}
	return simd.LoadUint8s(x)
}

func fillFloat32s(f func(i int) float32) simd.Float32s {
	l := simd.Float32s{}.Len()
	x := make([]float32, l, l)
	for i := range l {
		x[i] = f(i)
	}
	return simd.LoadFloat32s(x)
}

func stringFor(l int, f func(i int) int) string {
	pfx := "{"
	var want string
	for i := range l {
		want += pfx
		pfx = ","
		want += fmt.Sprintf("%d", f(i))
	}
	want += "}"
	return want
}

func TestToString(t *testing.T) {
	a := fillUint8s(func(i int) uint8 { return uint8(i) & 1 })
	b := fillUint8s(func(i int) uint8 { return uint8(i>>1) & 1 })
	m := a.Equal(b)
	wantM := stringFor(a.Len(),
		func(i int) int {
			if i&1 == (i>>1)&1 {
				return 1
			}
			return 0
		})
	if got := m.String(); wantM != got {
		t.Errorf("wantM=%s, got=%s", wantM, got)
	}
	wantA := stringFor(a.Len(),
		func(i int) int {
			return i & 1
		})
	if got := a.String(); wantA != got {
		t.Errorf("wantA=%s, got=%s", wantA, got)
	}

	f := fillFloat32s(func(i int) float32 { return float32(i) })
	wantF := stringFor(f.Len(),
		func(i int) int {
			return i
		})
	if got := f.String(); wantF != got {
		t.Errorf("wantF=%s, got=%s", wantF, got)
	}
}
