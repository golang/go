// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file

package main

import (
	"./p"
)

type T struct{ *p.S }
type I interface {
	get()
}

func main() {
	var t T
	p.F(t)
	var x interface{} = t
	_, ok := x.(I)
	if ok {
		panic("should not satisfy main.I")
	}
	_, ok = x.(p.I)
	if !ok {
		panic("should satisfy p.I")
	}
}
