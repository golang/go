// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type I interface{ foo() int }

type myint int

func (x myint) foo() int { return int(x) }

type myfloat float64

func (x myfloat) foo() int { return int(x) }

type myint32 int32

func (x myint32) foo() int { return int(x) }

func f[T I](i I) {
	switch x := i.(type) {
	case T, myint32:
		println("T/myint32", x.foo())
	default:
		println("other", x.foo())
	}
}
func main() {
	f[myfloat](myint(6))
	f[myfloat](myfloat(7))
	f[myfloat](myint32(8))
	f[myint32](myint32(9))
	f[myint](myint32(10))
	f[myint](myfloat(42))
}
