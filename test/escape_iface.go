// errorcheck -0 -m -l

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis for interface conversions.

package escape

var sink interface{}

type M interface {
	M()
}

func mescapes(m M) { // ERROR "leaking param: m"
	sink = m
}

func mdoesnotescape(m M) { // ERROR "m does not escape"
}

// Tests for type stored directly in iface and with value receiver method.
type M0 struct {
	p *int
}

func (M0) M() {
}

func efaceEscape0() {
	{
		i := 0
		v := M0{&i}
		var x M = v
		_ = x
	}
	{
		i := 0 // ERROR "moved to heap: i"
		v := M0{&i}
		var x M = v
		sink = x
	}
	{
		i := 0
		v := M0{&i}
		var x M = v
		v1 := x.(M0)
		_ = v1
	}
	{
		i := 0 // ERROR "moved to heap: i"
		v := M0{&i}
		// BAD: v does not escape to heap here
		var x M = v
		v1 := x.(M0)
		sink = v1
	}
	{
		i := 0
		v := M0{&i}
		var x M = v
		x.M() // ERROR "devirtualizing x.M"
	}
	{
		i := 0 // ERROR "moved to heap: i"
		v := M0{&i}
		var x M = v
		mescapes(x)
	}
	{
		i := 0
		v := M0{&i}
		var x M = v
		mdoesnotescape(x)
	}
}

// Tests for type stored indirectly in iface and with value receiver method.
type M1 struct {
	p *int
	x int
}

func (M1) M() {
}

func efaceEscape1() {
	{
		i := 0
		v := M1{&i, 0}
		var x M = v // ERROR "v does not escape"
		_ = x
	}
	{
		i := 0 // ERROR "moved to heap: i"
		v := M1{&i, 0}
		var x M = v // ERROR "v escapes to heap"
		sink = x
	}
	{
		i := 0
		v := M1{&i, 0}
		var x M = v // ERROR "v does not escape"
		v1 := x.(M1)
		_ = v1
	}
	{
		i := 0 // ERROR "moved to heap: i"
		v := M1{&i, 0}
		var x M = v // ERROR "v does not escape"
		v1 := x.(M1)
		sink = v1 // ERROR "v1 escapes to heap"
	}
	{
		i := 0
		v := M1{&i, 0}
		var x M = v // ERROR "v does not escape"
		x.M()       // ERROR "devirtualizing x.M"
	}
	{
		i := 0 // ERROR "moved to heap: i"
		v := M1{&i, 0}
		var x M = v // ERROR "v escapes to heap"
		mescapes(x)
	}
	{
		i := 0
		v := M1{&i, 0}
		var x M = v // ERROR "v does not escape"
		mdoesnotescape(x)
	}
}

// Tests for type stored directly in iface and with pointer receiver method.
type M2 struct {
	p *int
}

func (*M2) M() {
}

func efaceEscape2() {
	{
		i := 0
		v := &M2{&i} // ERROR "&M2{...} does not escape"
		var x M = v
		_ = x
	}
	{
		i := 0       // ERROR "moved to heap: i"
		v := &M2{&i} // ERROR "&M2{...} escapes to heap"
		var x M = v
		sink = x
	}
	{
		i := 0
		v := &M2{&i} // ERROR "&M2{...} does not escape"
		var x M = v
		v1 := x.(*M2)
		_ = v1
	}
	{
		i := 0       // ERROR "moved to heap: i"
		v := &M2{&i} // ERROR "&M2{...} escapes to heap"
		// BAD: v does not escape to heap here
		var x M = v
		v1 := x.(*M2)
		sink = v1
	}
	{
		i := 0       // ERROR "moved to heap: i"
		v := &M2{&i} // ERROR "&M2{...} does not escape"
		// BAD: v does not escape to heap here
		var x M = v
		v1 := x.(*M2)
		sink = *v1
	}
	{
		i := 0       // ERROR "moved to heap: i"
		v := &M2{&i} // ERROR "&M2{...} does not escape"
		// BAD: v does not escape to heap here
		var x M = v
		v1, ok := x.(*M2)
		sink = *v1
		_ = ok
	}
	{
		i := 0
		v := &M2{&i} // ERROR "&M2{...} does not escape"
		var x M = v
		x.M() // ERROR "devirtualizing x.M"
	}
	{
		i := 0       // ERROR "moved to heap: i"
		v := &M2{&i} // ERROR "&M2{...} escapes to heap"
		var x M = v
		mescapes(x)
	}
	{
		i := 0
		v := &M2{&i} // ERROR "&M2{...} does not escape"
		var x M = v
		mdoesnotescape(x)
	}
}

type T1 struct {
	p *int
}

type T2 struct {
	T1 T1
}

func dotTypeEscape() *T2 { // #11931
	var x interface{}
	x = &T1{p: new(int)} // ERROR "new\(int\) escapes to heap" "&T1{...} does not escape"
	return &T2{          // ERROR "&T2{...} escapes to heap"
		T1: *(x.(*T1)),
	}
}

func dotTypeEscape2() { // #13805, #15796
	{
		i := 0
		j := 0
		var v int
		var ok bool
		var x interface{} = i // ERROR "0 does not escape"
		var y interface{} = j // ERROR "0 does not escape"

		*(&v) = x.(int)
		*(&v), *(&ok) = y.(int)
	}
	{ // #13805, #15796
		i := 0
		j := 0
		var ok bool
		var x interface{} = i // ERROR "0 does not escape"
		var y interface{} = j // ERROR "0 does not escape"

		sink = x.(int)         // ERROR "x.\(int\) escapes to heap"
		sink, *(&ok) = y.(int) // ERROR "autotmp_.* escapes to heap"
	}
	{
		i := 0 // ERROR "moved to heap: i"
		j := 0 // ERROR "moved to heap: j"
		var ok bool
		var x interface{} = &i
		var y interface{} = &j

		sink = x.(*int)
		sink, *(&ok) = y.(*int)
	}
}

func issue42279() {
	type I interface{ M() }
	type T struct{ I }

	var i I = T{} // ERROR "T\{\} does not escape"
	i.M()         // ERROR "partially devirtualizing i.M to T"
}
