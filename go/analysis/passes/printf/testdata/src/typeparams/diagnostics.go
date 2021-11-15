// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package typeparams

import "fmt"

func TestBasicTypeParams[T interface{ ~int }, E error, F fmt.Formatter, S fmt.Stringer, A any](t T, e E, f F, s S, a A) {
	fmt.Printf("%d", t)
	fmt.Printf("%s", t) // want "wrong type.*contains ~int"
	fmt.Printf("%v", t)
	fmt.Printf("%d", e) // want "wrong type"
	fmt.Printf("%s", e)
	fmt.Errorf("%w", e)
	fmt.Printf("%a", f)
	fmt.Printf("%d", f)
	fmt.Printf("%T", f.Format)
	fmt.Printf("%p", f.Format)
	fmt.Printf("%s", s)
	fmt.Errorf("%w", s) // want "wrong type"
	fmt.Printf("%d", a) // want "wrong type"
	fmt.Printf("%s", a) // want "wrong type"
	fmt.Printf("%v", a)
	fmt.Printf("%T", a)
}

type Constraint interface {
	~int
}

func TestNamedConstraints_Issue49597[T Constraint](t T) {
	fmt.Printf("%d", t)
	fmt.Printf("%s", t) // want "wrong type.*contains ~int"
}

func TestNestedTypeParams[T interface{ ~int }, S interface{ ~string }]() {
	var x struct {
		f int
		t T
	}
	fmt.Printf("%d", x)
	fmt.Printf("%s", x) // want "wrong type"
	var y struct {
		f string
		t S
	}
	fmt.Printf("%d", y) // want "wrong type"
	fmt.Printf("%s", y)
	var m1 map[T]T
	fmt.Printf("%d", m1)
	fmt.Printf("%s", m1) // want "wrong type"
	var m2 map[S]S
	fmt.Printf("%d", m2) // want "wrong type"
	fmt.Printf("%s", m2)
}

type R struct {
	F []R
}

func TestRecursiveTypeDefinition() {
	var r []R
	fmt.Printf("%d", r) // No error: avoids infinite recursion.
}

func TestRecursiveTypeParams[T1 ~[]T2, T2 ~[]T1 | string, T3 ~struct{ F T3 }](t1 T1, t2 T2, t3 T3) {
	// No error is reported on the following lines to avoid infinite recursion.
	fmt.Printf("%s", t1)
	fmt.Printf("%s", t2)
	fmt.Printf("%s", t3)
}

func TestRecusivePointers[T1 ~*T2, T2 ~*T1](t1 T1, t2 T2) {
	// No error: we can't determine if pointer rules apply.
	fmt.Printf("%s", t1)
	fmt.Printf("%s", t2)
}

func TestEmptyTypeSet[T interface {
	int | string
	float64
}](t T) {
	fmt.Printf("%s", t) // No error: empty type set.
}

func TestPointerRules[T ~*[]int | *[2]int](t T) {
	var slicePtr *[]int
	var arrayPtr *[2]int
	fmt.Printf("%d", slicePtr)
	fmt.Printf("%d", arrayPtr)
	fmt.Printf("%d", t)
}

func TestInterfacePromotion[E interface {
	~int
	Error() string
}, S interface {
	float64
	String() string
}](e E, s S) {
	fmt.Printf("%d", e)
	fmt.Printf("%s", e)
	fmt.Errorf("%w", e)
	fmt.Printf("%d", s) // want "wrong type.*contains float64"
	fmt.Printf("%s", s)
	fmt.Errorf("%w", s) // want "wrong type"
}

type myInt int

func TestTermReduction[T1 interface{ ~int | string }, T2 interface {
	~int | string
	myInt
}](t1 T1, t2 T2) {
	fmt.Printf("%d", t1) // want "wrong type.*contains string"
	fmt.Printf("%s", t1) // want "wrong type.*contains ~int"
	fmt.Printf("%d", t2)
	fmt.Printf("%s", t2) // want "wrong type.*contains typeparams.myInt"
}
