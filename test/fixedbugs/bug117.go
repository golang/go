// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type S struct {
	a int
}
type PS *S

func (p *S) get() int {
	return p.a
}

func fn(p PS) int {
	// p has type PS, and PS has no methods.
	// (a compiler might see that p is a pointer
	// and go looking in S without noticing PS.)
	return p.get() // ERROR "undefined"
}
func main() {
	s := S{1}
	if s.get() != 1 {
		panic("fail")
	}
}
