// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

type Element interface {
}

type Vector struct {
	nelem int
	elem  []Element
}

func New() *Vector {
	v := new(Vector)
	v.nelem = 0
	v.elem = make([]Element, 10)
	return v
}

func (v *Vector) At(i int) Element {
	return v.elem[i]
}

func (v *Vector) Insert(e Element) {
	v.elem[v.nelem] = e
	v.nelem++
}

func main() {
	type I struct{ val int }
	i0 := new(I)
	i0.val = 0
	i1 := new(I)
	i1.val = 11
	i2 := new(I)
	i2.val = 222
	i3 := new(I)
	i3.val = 3333
	i4 := new(I)
	i4.val = 44444
	v := New()
	r := "hi\n"
	v.Insert(i4)
	v.Insert(i3)
	v.Insert(i2)
	v.Insert(i1)
	v.Insert(i0)
	for i := 0; i < v.nelem; i++ {
		var x *I
		x = v.At(i).(*I)
		r += fmt.Sprintln(i, x.val) // prints correct list
	}
	for i := 0; i < v.nelem; i++ {
		r += fmt.Sprintln(i, v.At(i).(*I).val)
	}
	expect := `hi
0 44444
1 3333
2 222
3 11
4 0
0 44444
1 3333
2 222
3 11
4 0
`
	if r != expect {
		panic(r)
	}
}

/*
bug027.go:50: illegal types for operand
	(<Element>I{}) CONV (<I>{})
bug027.go:50: illegal types for operand
	(<Element>I{}) CONV (<I>{})
*/
