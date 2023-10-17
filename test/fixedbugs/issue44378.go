// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test case caused a panic in the compiler's DWARF gen code.

// Note to future maintainers of this code:
//
//    ** Do NOT run gofmt when editing this file **
//
// In order for the buggy behavior to be triggered in the compiler,
// we need to have a the function of interest all on one gigantic line.

package a

type O interface{}
type IO int
type OS int

type A struct {
	x int
}

// original versions of the two function
func (p *A) UO(o O) {
	p.r(o, o)
}
func (p *A) r(o1, o2 O) {
	switch x := o1.(type) {
	case *IO:
		p.x = int(*x)
	case *OS:
		p.x = int(*x + 2)
	}
}

// see note above about the importance of all this code winding up on one line.
var myverylongname0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789 int ; func (p *A) UO2(o O) { p.r2(o, o); }; func (p *A) r2(o1, o2 O) { switch x := o1.(type) { case *IO:	p.x = int(*x); 	case *OS: 	p.x = int(*x + 2); } }
