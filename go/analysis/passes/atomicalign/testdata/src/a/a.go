// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the atomic alignment checker.

//go:build arm || 386
// +build arm 386

package testdata

import (
	"io"
	"sync/atomic"
)

func intsAlignment() {
	var s struct {
		a bool
		b uint8
		c int8
		d byte
		f int16
		g int16
		h int64
		i byte
		j uint64
	}
	atomic.AddInt64(&s.h, 9)
	atomic.AddUint64(&s.j, 0) // want "address of non 64-bit aligned field .j passed to atomic.AddUint64"
}

func floatAlignment() {
	var s struct {
		a float32
		b int64
		c float32
		d float64
		e uint64
	}
	atomic.LoadInt64(&s.b) // want "address of non 64-bit aligned field .b passed to atomic.LoadInt64"
	atomic.LoadUint64(&s.e)
}

func uintptrAlignment() {
	var s struct {
		a uintptr
		b int64
		c int
		d uint
		e int32
		f uint64
	}
	atomic.StoreInt64(&s.b, 0) // want "address of non 64-bit aligned field .b passed to atomic.StoreInt64"
	atomic.StoreUint64(&s.f, 0)
}

func runeAlignment() {
	var s struct {
		a rune
		b int64
		_ rune
		c uint64
	}
	atomic.SwapInt64(&s.b, 0) // want "address of non 64-bit aligned field .b passed to atomic.SwapInt64"
	atomic.SwapUint64(&s.c, 0)
}

func complexAlignment() {
	var s struct {
		a complex64
		b int64
		c complex128
		d uint64
	}
	atomic.CompareAndSwapInt64(&s.b, 0, 1)
	atomic.CompareAndSwapUint64(&s.d, 0, 1)
}

// continuer ici avec les tests

func channelAlignment() {
	var a struct {
		a chan struct{}
		b int64
		c <-chan struct{}
		d uint64
	}

	atomic.AddInt64(&a.b, 0) // want "address of non 64-bit aligned field .b passed to atomic.AddInt64"
	atomic.AddUint64(&a.d, 0)
}

func arrayAlignment() {
	var a struct {
		a [1]uint16
		b int64
		_ [2]uint16
		c int64
		d [1]uint16
		e uint64
	}

	atomic.LoadInt64(&a.b) // want "address of non 64-bit aligned field .b passed to atomic.LoadInt64"
	atomic.LoadInt64(&a.c)
	atomic.LoadUint64(&a.e)   // want "address of non 64-bit aligned field .e passed to atomic.LoadUint64"
	(atomic.LoadUint64)(&a.e) // want "address of non 64-bit aligned field .e passed to atomic.LoadUint64"
}

func anonymousFieldAlignment() {
	var f struct {
		a, b int32
		c, d int64
		_    bool
		e, f uint64
	}

	atomic.StoreInt64(&f.c, 12)
	atomic.StoreInt64(&f.d, 27)
	atomic.StoreUint64(&f.e, 6)  // want "address of non 64-bit aligned field .e passed to atomic.StoreUint64"
	atomic.StoreUint64(&f.f, 79) // want "address of non 64-bit aligned field .f passed to atomic.StoreUint64"
}

type ts struct {
	e  int64
	e2 []int
	f  uint64
}

func typedStructAlignment() {
	var b ts
	atomic.SwapInt64(&b.e, 9)
	atomic.SwapUint64(&b.f, 9) // want "address of non 64-bit aligned field .f passed to atomic.SwapUint64"
}

func aliasAlignment() {
	type (
		mybytea uint8
		mybyteb byte
		mybytec = uint8
		mybyted = byte
	)

	var e struct {
		a    byte
		b    mybytea
		c    mybyteb
		e    mybytec
		f    int64
		g, h uint16
		i    uint64
	}

	atomic.CompareAndSwapInt64(&e.f, 0, 1) // want "address of non 64-bit aligned field .f passed to atomic.CompareAndSwapInt64"
	atomic.CompareAndSwapUint64(&e.i, 1, 2)
}

func stringAlignment() {
	var a struct {
		a uint32
		b string
		c int64
	}
	atomic.AddInt64(&a.c, 10) // want "address of non 64-bit aligned field .c passed to atomic.AddInt64"
}

func sliceAlignment() {
	var s struct {
		a []int32
		b int64
		c uint32
		d uint64
	}

	atomic.LoadInt64(&s.b) // want "address of non 64-bit aligned field .b passed to atomic.LoadInt64"
	atomic.LoadUint64(&s.d)
}

func interfaceAlignment() {
	var s struct {
		a interface{}
		b int64
		c io.Writer
		e int64
		_ int32
		f uint64
	}

	atomic.StoreInt64(&s.b, 9)
	atomic.StoreInt64(&s.e, 9)
	atomic.StoreUint64(&s.f, 9) // want "address of non 64-bit aligned field .f passed to atomic.StoreUint64"
}

func pointerAlignment() {
	var s struct {
		a, b *int
		c    int64
		d    *interface{}
		e    uint64
	}

	atomic.SwapInt64(&s.c, 9)
	atomic.SwapUint64(&s.e, 9) // want "address of non 64-bit aligned field .e passed to atomic.SwapUint64"
}

// non-struct fields are already 64-bits correctly aligned per Go spec
func nonStructFields() {
	var (
		a *int64
		b [2]uint64
		c int64
	)

	atomic.CompareAndSwapInt64(a, 10, 11)
	atomic.CompareAndSwapUint64(&b[0], 5, 23)
	atomic.CompareAndSwapInt64(&c, -1, -15)
}

func embeddedStructFields() {
	var s1 struct {
		_ struct{ _ int32 }
		a int64
		_ struct{}
		b uint64
		_ struct{ _ [2]uint16 }
		c int64
	}

	atomic.AddInt64(&s1.a, 9)  // want "address of non 64-bit aligned field .a passed to atomic.AddInt64"
	atomic.AddUint64(&s1.b, 9) // want "address of non 64-bit aligned field .b passed to atomic.AddUint64"
	atomic.AddInt64(&s1.c, 9)
}
