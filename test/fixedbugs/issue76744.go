// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Converting a constant to a type parameter and then to an interface used
// to point into runtime.staticuint64s, at the last byte of an element on a
// big-endian platform. Reading the value back out was then a load from an
// odd offset off the symbol, which the s390x linker cannot encode.
//
// CL 786920 fixed this incidentally, by making a constant converted to a
// type parameter non-constant, as the spec requires.

package main

type I interface{ bool | float64 }

type G[T I] interface{ Val() T }

type BoolLit bool

func (b BoolLit) Val() bool { return bool(b) }

type F64Lit float64

func (f F64Lit) Val() float64 { return float64(f) }

func newLit[T I](val T) any {
	switch v := any(val).(type) {
	case bool:
		return BoolLit(v)
	case float64:
		return F64Lit(v)
	}
	panic("unreachable")
}

func f[T I]() T {
	var z T
	return newLit(z).(G[T]).Val()
}

func main() {
	if got := f[bool](); got != false {
		panic("unexpected result")
	}
	if got := f[float64](); got != 0 {
		panic("unexpected result")
	}
}
