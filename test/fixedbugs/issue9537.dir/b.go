// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"

	"./a"
)

type X struct {
	*a.X
}

type Intf interface {
	Get()        []byte
	RetPtr(int)  *int
	RetRPtr(int) (int, *int)
}

func main() {
	x := &a.X{T: [32]byte{1, 2, 3, 4}}
	var ix Intf = X{x}
	t1 := ix.Get()
	t2 := x.Get()
	if !bytes.Equal(t1, t2) {
		panic(t1)
	}

	p1 := ix.RetPtr(5)
	p2 := x.RetPtr(7)
	if *p1 != 6 || *p2 != 8 {
		panic(*p1)
	}

	r1, r2 := ix.RetRPtr(10)
	r3, r4 := x.RetRPtr(13)
	if r1 != 11 || *r2 != 11 || r3 != 14 || *r4 != 14 {
		panic("bad RetRPtr")
	}
}
