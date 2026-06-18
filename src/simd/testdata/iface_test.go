// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package testdata_test

import (
	"reflect"
	"simd"
	"simd/testdata/iface"
	"testing"
)

func TestIFaceFoo(t *testing.T) {
	u := simd.BroadcastFloat32s(4)
	v := simd.BroadcastFloat32s(1)
	vc := iface.VC(v)
	uc := iface.VC(u)

	hv := iface.MakeHasFoo(vc) // generic interface w/ Foo method

	sum := hv.Foo(uc)

	s := make([]float32, u.Len())

	sum.Data().Store(s) // The method of a dependent type works.

	if s[0] != 5 {
		t.Errorf("(from Data()) expected 5, got %f", s[0])
	}

	sum.Field.Store(s)

	if s[0] != 5 {
		t.Errorf("(from Field) expected 5, got %f", s[0])
	}
}

func TestIFaceBar(t *testing.T) {
	u := simd.BroadcastFloat32s(4)
	v := simd.BroadcastFloat32s(1)
	vc := iface.VC(v)
	uc := iface.VC(u)

	hv := iface.MakeHasBar(vc) // non-generic interface w/ Foo method

	sum := hv.Bar(uc)

	s := make([]float32, u.Len())

	sum.Data().Store(s) // The method of a dependent type works.

	if s[0] != 5 {
		t.Errorf("(from Data()) expected 5, got %f", s[0])
	}

	sum.Field.Store(s)

	if s[0] != 5 {
		t.Errorf("(from Field) expected 5, got %f", s[0])
	}
}

func TestIFaceEmbedFoo(t *testing.T) {
	u := simd.BroadcastFloat32s(4)
	v := simd.BroadcastFloat32s(1)
	vc := iface.VC(v)
	uc := iface.VC(u)

	hv := iface.MakeHasEmbedFoo(vc) // generic interface w/ Foo method

	sum := hv.Foo(uc)

	s := make([]float32, u.Len())

	sum.Data().Store(s) // The method of a dependent type works.

	if s[0] != 5 {
		t.Errorf("(from Data()) expected 5, got %f", s[0])
	}

	sum.Field.Store(s)

	if s[0] != 5 {
		t.Errorf("(from Field) expected 5, got %f", s[0])
	}

	rv := reflect.ValueOf(hv)
	rt := rv.Type()

	t.Logf("reflect.value is %v", rv)
	t.Logf("reflect.type is %v", rt)
}

func TestIFaceEmbedBar(t *testing.T) {
	u := simd.BroadcastFloat32s(4)
	v := simd.BroadcastFloat32s(1)
	vc := iface.VC(v)
	uc := iface.VC(u)

	hv := iface.MakeHasEmbedBar(vc) // generic interface w/ Foo method

	sum := hv.Bar(uc)

	s := make([]float32, u.Len())

	sum.Data().Store(s) // The method of a dependent type works.

	if s[0] != 5 {
		t.Errorf("(from Data()) expected 5, got %f", s[0])
	}

	sum.Field.Store(s)

	if s[0] != 5 {
		t.Errorf("(from Field) expected 5, got %f", s[0])
	}

	rv := reflect.ValueOf(hv)
	rt := rv.Type()

	t.Logf("reflect.value is %v", rv)
	t.Logf("reflect.type is %v", rt)
}

func TestIFaceVL(t *testing.T) {
	var v simd.Int8s
	if a, b := iface.VL, iface.Generic[simd.Int8s](1); a != b {
		t.Errorf("expected iface.VL [%d] == iface.Generic[simd.Int8s](1) [%d], but not true", a, b)
	}
	if a, b := iface.VL, v.Len()+1; a != b {
		t.Errorf("expected iface.VL [%d] == v.Len()+1 [%d], but not true", a, b)
	}
}
