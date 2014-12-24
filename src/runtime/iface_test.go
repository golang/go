// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"testing"
)

type I1 interface {
	Method1()
}

type I2 interface {
	Method1()
	Method2()
}

type TS uint16
type TM uintptr
type TL [2]uintptr

func (TS) Method1() {}
func (TS) Method2() {}
func (TM) Method1() {}
func (TM) Method2() {}
func (TL) Method1() {}
func (TL) Method2() {}

var (
	e  interface{}
	e_ interface{}
	i1 I1
	i2 I2
	ts TS
	tm TM
	tl TL
)

// Issue 9370
func TestCmpIfaceConcreteAlloc(t *testing.T) {
	if runtime.Compiler != "gc" {
		t.Skip("skipping on non-gc compiler")
	}

	n := testing.AllocsPerRun(1, func() {
		_ = e == ts
		_ = i1 == ts
		_ = e == 1
	})

	if n > 0 {
		t.Fatalf("iface cmp allocs=%v; want 0", n)
	}
}

func BenchmarkEqEfaceConcrete(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = e == ts
	}
}

func BenchmarkEqIfaceConcrete(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = i1 == ts
	}
}

func BenchmarkNeEfaceConcrete(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = e != ts
	}
}

func BenchmarkNeIfaceConcrete(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = i1 != ts
	}
}

func BenchmarkConvT2ESmall(b *testing.B) {
	for i := 0; i < b.N; i++ {
		e = ts
	}
}

func BenchmarkConvT2EUintptr(b *testing.B) {
	for i := 0; i < b.N; i++ {
		e = tm
	}
}

func BenchmarkConvT2ELarge(b *testing.B) {
	for i := 0; i < b.N; i++ {
		e = tl
	}
}

func BenchmarkConvT2ISmall(b *testing.B) {
	for i := 0; i < b.N; i++ {
		i1 = ts
	}
}

func BenchmarkConvT2IUintptr(b *testing.B) {
	for i := 0; i < b.N; i++ {
		i1 = tm
	}
}

func BenchmarkConvT2ILarge(b *testing.B) {
	for i := 0; i < b.N; i++ {
		i1 = tl
	}
}

func BenchmarkConvI2E(b *testing.B) {
	i2 = tm
	for i := 0; i < b.N; i++ {
		e = i2
	}
}

func BenchmarkConvI2I(b *testing.B) {
	i2 = tm
	for i := 0; i < b.N; i++ {
		i1 = i2
	}
}

func BenchmarkAssertE2T(b *testing.B) {
	e = tm
	for i := 0; i < b.N; i++ {
		tm = e.(TM)
	}
}

func BenchmarkAssertE2TLarge(b *testing.B) {
	e = tl
	for i := 0; i < b.N; i++ {
		tl = e.(TL)
	}
}

func BenchmarkAssertE2I(b *testing.B) {
	e = tm
	for i := 0; i < b.N; i++ {
		i1 = e.(I1)
	}
}

func BenchmarkAssertI2T(b *testing.B) {
	i1 = tm
	for i := 0; i < b.N; i++ {
		tm = i1.(TM)
	}
}

func BenchmarkAssertI2I(b *testing.B) {
	i1 = tm
	for i := 0; i < b.N; i++ {
		i2 = i1.(I2)
	}
}

func BenchmarkAssertI2E(b *testing.B) {
	i1 = tm
	for i := 0; i < b.N; i++ {
		e = i1.(interface{})
	}
}

func BenchmarkAssertE2E(b *testing.B) {
	e = tm
	for i := 0; i < b.N; i++ {
		e_ = e
	}
}
