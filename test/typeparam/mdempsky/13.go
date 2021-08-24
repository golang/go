// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Interface which will be used as a regular interface type and as a type bound.
type Mer interface{
	M()
}

// Interface that is a superset of Mer.
type Mer2 interface {
	M()
	String() string
}

func F[T Mer](t T) {
	T.M(t)
	t.M()
}

type MyMer int

func (MyMer) M() {}
func (MyMer) String() string {
	return "aa"
}

// Parameterized interface
type Abs[T any] interface {
	Abs() T
}

func G[T Abs[U], U any](t T) {
	T.Abs(t)
	t.Abs()
}

type MyInt int
func (m MyInt) Abs() MyInt {
	if m < 0 {
		return -m
	}
	return m
}

type Abs2 interface {
	Abs() MyInt
}


func main() {
	mm := MyMer(3)
	ms := struct{ Mer }{Mer: mm }

	// Testing F with an interface type arg: Mer and Mer2
	F[Mer](mm)
	F[Mer2](mm)
	F[struct{ Mer }](ms)
	F[*struct{ Mer }](&ms)

	ms2 := struct { MyMer }{MyMer: mm}
	ms3 := struct { *MyMer }{MyMer: &mm}

	// Testing F with a concrete type arg
	F[MyMer](mm)
	F[*MyMer](&mm)
	F[struct{ MyMer }](ms2)
	F[struct{ *MyMer }](ms3)
	F[*struct{ MyMer }](&ms2)
	F[*struct{ *MyMer }](&ms3)

	// Testing G with a concrete type args
	mi := MyInt(-3)
	G[MyInt,MyInt](mi)

	// Interface Abs[MyInt] holding an mi.
	intMi := Abs[MyInt](mi)
	// First type arg here is Abs[MyInt], an interface type.
	G[Abs[MyInt],MyInt](intMi)
}
