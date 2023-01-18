// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f1[P int | string]()            {}
func f2[P ~int | string | float64]() {}
func f3[P int](x P)                  {}

type myInt int
type myFloat float64

func _() {
	_ = f1[int]
	_ = f1[myInt /* ERROR "possibly missing ~ for int in int | string" */]
	_ = f2[myInt]
	_ = f2[myFloat /* ERROR "possibly missing ~ for float64 in ~int | string | float64" */]
	var x myInt
	f3 /* ERROR "myInt does not satisfy int (possibly missing ~ for int in int)" */ (x)
}

// test case from the issue

type SliceConstraint[T any] interface {
	[]T
}

func Map[S SliceConstraint[E], E any](s S, f func(E) E) S {
	return s
}

type MySlice []int

func f(s MySlice) {
	Map[MySlice /* ERROR "MySlice does not satisfy SliceConstraint[int] (possibly missing ~ for []int in SliceConstraint[int])" */, int](s, nil)
}
