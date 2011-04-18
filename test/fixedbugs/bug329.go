// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Value struct {
	X interface{}
	Y int
}

type Struct struct {
	X complex128
}

const magic = 1 + 2i

func (Value) Complex(x complex128) {
	if x != magic {
		println(x)
		panic("bad complex magic")
	}
}

func f(x *byte, y, z int) complex128 {
	return magic
}

func (Value) Struct(x Struct) {
	if x.X != magic {
		println(x.X)
		panic("bad struct magic")
	}
}

func f1(x *byte, y, z int) Struct {
	return Struct{magic}
}

func main() {
	var v Value
	v.Struct(f1(nil, 0, 0)) // ok
	v.Complex(f(nil, 0, 0)) // used to fail
}
