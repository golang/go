// $G $D/$F.go && $L $F.$A && ./$A.out || echo BUG: bug285

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test for issue 778: Map key values that are assignment
// compatible with the map key type must be accepted according
// to the spec: http://golang.org/doc/go_spec.html#Indexes .

package main

type T2 struct {
	x int
}

func (t *T2) f() int { return t.x }

func main() {
	type B bool
	b := B(false)
	mb := make(map[B]int)
	mb[false] = 42 // this should work: false is assignment compatible with B
	mb[b] = 42

	type Z int
	z := Z(0)
	mz := make(map[Z]int)
	mz[0] = 42
	mz[z] = 42

	type S string
	s := S("foo")
	ms := make(map[S]int)
	ms["foo"] = 42
	ms[s] = 42

	type T struct {
		x int
	}
	type P *T
	p := P(nil)
	mp := make(map[P]int)
	mp[nil] = 42
	mp[p] = 42
	mp[&T{7}] = 42

	type C chan int
	c := make(C)
	mc := make(map[C]int)
	mc[nil] = 42
	mc[c] = 42
	mc[make(C)] = 42

	type I1 interface{}
	type I2 interface {
		f() int
	}
	var i0 interface{} = z
	var i1 I1 = p
	m0 := make(map[interface{}]int)
	m1 := make(map[I1]int)
	m2 := make(map[I2]int)
	m0[i0] = 42
	m0[i1] = 42
	m0[z] = 42 // this should work: z is assignment-compatible with interface{}
	m0[new(struct {
		x int
	})] = 42       // this should work: *struct{x int} is assignment-compatible with interface{}
	m0[p] = 42     // this should work: p is assignment-compatible with interface{}
	m0[false] = 42 // this should work: false is assignment-compatible with interface{}
	m0[17] = 42    // this should work: 17 is assignment-compatible with interface{}
	m0["foo"] = 42 // this should work: "foo" is assignment-compatible with interface{}

	m1[i0] = 42
	m1[i1] = 42
	m1[new(struct {
		x int
	})] = 42       // this should work: *struct{x int} is assignment-compatible with I1
	m1[false] = 42 // this should work: false is assignment-compatible with I1
	m1[17] = 42    // this should work: 17 is assignment-compatible with I1
	m1["foo"] = 42 // this should work: "foo" is assignment-compatible with I1

	m2[new(T2)] = 42 // this should work: *T2 is assignment-compatible with I2
}

/*
6g -e bug286.go
bug286.go:23: invalid map index false - need type B
bug286.go:80: invalid map index z - need type interface { }
bug286.go:83: invalid map index new(struct { x int }) - need type interface { }
bug286.go:84: invalid map index p - need type interface { }
bug286.go:85: invalid map index false - need type interface { }
bug286.go:86: invalid map index 17 - need type interface { }
bug286.go:87: invalid map index "foo" - need type interface { }
bug286.go:93: invalid map index new(struct { x int }) - need type I1
bug286.go:94: invalid map index false - need type I1
bug286.go:95: invalid map index 17 - need type I1
bug286.go:96: invalid map index "foo" - need type I1
bug286.go:99: invalid map index new(T2) - need type I2
bug286.go:100: invalid map index t2 - need type I2
*/
