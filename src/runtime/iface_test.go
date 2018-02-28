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

type T8 uint8
type T16 uint16
type T32 uint32
type T64 uint64
type Tstr string
type Tslice []byte

func (T8) Method1()     {}
func (T16) Method1()    {}
func (T32) Method1()    {}
func (T64) Method1()    {}
func (Tstr) Method1()   {}
func (Tslice) Method1() {}

var (
	e  interface{}
	e_ interface{}
	i1 I1
	i2 I2
	ts TS
	tm TM
	tl TL
	ok bool
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

func BenchmarkAssertE2T2(b *testing.B) {
	e = tm
	for i := 0; i < b.N; i++ {
		tm, ok = e.(TM)
	}
}

func BenchmarkAssertE2T2Blank(b *testing.B) {
	e = tm
	for i := 0; i < b.N; i++ {
		_, ok = e.(TM)
	}
}

func BenchmarkAssertI2E2(b *testing.B) {
	i1 = tm
	for i := 0; i < b.N; i++ {
		e, ok = i1.(interface{})
	}
}

func BenchmarkAssertI2E2Blank(b *testing.B) {
	i1 = tm
	for i := 0; i < b.N; i++ {
		_, ok = i1.(interface{})
	}
}

func BenchmarkAssertE2E2(b *testing.B) {
	e = tm
	for i := 0; i < b.N; i++ {
		e_, ok = e.(interface{})
	}
}

func BenchmarkAssertE2E2Blank(b *testing.B) {
	e = tm
	for i := 0; i < b.N; i++ {
		_, ok = e.(interface{})
	}
}

func TestNonEscapingConvT2E(t *testing.T) {
	m := make(map[interface{}]bool)
	m[42] = true
	if !m[42] {
		t.Fatalf("42 is not present in the map")
	}
	if m[0] {
		t.Fatalf("0 is present in the map")
	}

	n := testing.AllocsPerRun(1000, func() {
		if m[0] {
			t.Fatalf("0 is present in the map")
		}
	})
	if n != 0 {
		t.Fatalf("want 0 allocs, got %v", n)
	}
}

func TestNonEscapingConvT2I(t *testing.T) {
	m := make(map[I1]bool)
	m[TM(42)] = true
	if !m[TM(42)] {
		t.Fatalf("42 is not present in the map")
	}
	if m[TM(0)] {
		t.Fatalf("0 is present in the map")
	}

	n := testing.AllocsPerRun(1000, func() {
		if m[TM(0)] {
			t.Fatalf("0 is present in the map")
		}
	})
	if n != 0 {
		t.Fatalf("want 0 allocs, got %v", n)
	}
}

func TestZeroConvT2x(t *testing.T) {
	tests := []struct {
		name string
		fn   func()
	}{
		{name: "E8", fn: func() { e = eight8 }},  // any byte-sized value does not allocate
		{name: "E16", fn: func() { e = zero16 }}, // zero values do not allocate
		{name: "E32", fn: func() { e = zero32 }},
		{name: "E64", fn: func() { e = zero64 }},
		{name: "Estr", fn: func() { e = zerostr }},
		{name: "Eslice", fn: func() { e = zeroslice }},
		{name: "Econstflt", fn: func() { e = 99.0 }}, // constants do not allocate
		{name: "Econststr", fn: func() { e = "change" }},
		{name: "I8", fn: func() { i1 = eight8I }},
		{name: "I16", fn: func() { i1 = zero16I }},
		{name: "I32", fn: func() { i1 = zero32I }},
		{name: "I64", fn: func() { i1 = zero64I }},
		{name: "Istr", fn: func() { i1 = zerostrI }},
		{name: "Islice", fn: func() { i1 = zerosliceI }},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			n := testing.AllocsPerRun(1000, test.fn)
			if n != 0 {
				t.Errorf("want zero allocs, got %v", n)
			}
		})
	}
}

var (
	eight8  uint8 = 8
	eight8I T8    = 8

	zero16  uint16 = 0
	zero16I T16    = 0
	one16   uint16 = 1

	zero32  uint32 = 0
	zero32I T32    = 0
	one32   uint32 = 1

	zero64  uint64 = 0
	zero64I T64    = 0
	one64   uint64 = 1

	zerostr  string = ""
	zerostrI Tstr   = ""
	nzstr    string = "abc"

	zeroslice  []byte = nil
	zerosliceI Tslice = nil
	nzslice    []byte = []byte("abc")

	zerobig [512]byte
	nzbig   [512]byte = [512]byte{511: 1}
)

func BenchmarkConvT2Ezero(b *testing.B) {
	b.Run("zero", func(b *testing.B) {
		b.Run("16", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				e = zero16
			}
		})
		b.Run("32", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				e = zero32
			}
		})
		b.Run("64", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				e = zero64
			}
		})
		b.Run("str", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				e = zerostr
			}
		})
		b.Run("slice", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				e = zeroslice
			}
		})
		b.Run("big", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				e = zerobig
			}
		})
	})
	b.Run("nonzero", func(b *testing.B) {
		b.Run("16", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				e = one16
			}
		})
		b.Run("32", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				e = one32
			}
		})
		b.Run("64", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				e = one64
			}
		})
		b.Run("str", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				e = nzstr
			}
		})
		b.Run("slice", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				e = nzslice
			}
		})
		b.Run("big", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				e = nzbig
			}
		})
	})
}
