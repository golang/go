// run

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The gofrontend used to mishandle this code due to a pass ordering issue.
// It was inconsistent as to whether unsafe.Sizeof(byte(0)) was a constant,
// and therefore as to whether it was a direct-iface type.

package main

import "unsafe"

type A [unsafe.Sizeof(byte(0))]*byte

func (r A) V() byte {
	return *r[0]
}

func F() byte {
	panic("F") // should never be called
}

type B [unsafe.Sizeof(F())]*byte

func (r B) V() byte {
	return *r[0]
}

func main() {
	b := byte(1)
	v := A{&b}.V() + B{&b}.V()
	if v != 2 {
		panic(v)
	}
}
