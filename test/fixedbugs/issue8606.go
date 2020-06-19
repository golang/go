// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check to make sure that we compare fields in order. See issue 8606.

package main

import "fmt"

func main() {
	type A [2]interface{}
	type A2 [6]interface{}
	type S struct{ x, y interface{} }
	type S2 struct{ x, y, z, a, b, c interface{} }
	type T1 struct {
		i interface{}
		a int64
		j interface{}
	}
	type T2 struct {
		i       interface{}
		a, b, c int64
		j       interface{}
	}
	type T3 struct {
		i interface{}
		s string
		j interface{}
	}
	b := []byte{1}

	for _, test := range []struct {
		panic bool
		a, b  interface{}
	}{
		{false, A{1, b}, A{2, b}},
		{true, A{b, 1}, A{b, 2}},
		{false, A{1, b}, A{"2", b}},
		{true, A{b, 1}, A{b, "2"}},

		{false, A2{1, b}, A2{2, b}},
		{true, A2{b, 1}, A2{b, 2}},
		{false, A2{1, b}, A2{"2", b}},
		{true, A2{b, 1}, A2{b, "2"}},

		{false, S{1, b}, S{2, b}},
		{true, S{b, 1}, S{b, 2}},
		{false, S{1, b}, S{"2", b}},
		{true, S{b, 1}, S{b, "2"}},

		{false, S2{x: 1, y: b}, S2{x: 2, y: b}},
		{true, S2{x: b, y: 1}, S2{x: b, y: 2}},
		{false, S2{x: 1, y: b}, S2{x: "2", y: b}},
		{true, S2{x: b, y: 1}, S2{x: b, y: "2"}},

		{true, T1{i: b, a: 1}, T1{i: b, a: 2}},
		{false, T1{a: 1, j: b}, T1{a: 2, j: b}},
		{true, T2{i: b, a: 1}, T2{i: b, a: 2}},
		{false, T2{a: 1, j: b}, T2{a: 2, j: b}},
		{true, T3{i: b, s: "foo"}, T3{i: b, s: "bar"}},
		{false, T3{s: "foo", j: b}, T3{s: "bar", j: b}},
		{true, T3{i: b, s: "fooz"}, T3{i: b, s: "bar"}},
		{false, T3{s: "fooz", j: b}, T3{s: "bar", j: b}},
	} {
		f := func() {
			defer func() {
				if recover() != nil {
					panic(fmt.Sprintf("comparing %#v and %#v panicked", test.a, test.b))
				}
			}()
			if test.a == test.b {
				panic(fmt.Sprintf("values %#v and %#v should not be equal", test.a, test.b))
			}
		}
		if test.panic {
			shouldPanic(fmt.Sprintf("comparing %#v and %#v did not panic", test.a, test.b), f)
		} else {
			f() // should not panic
		}
	}
}

func shouldPanic(name string, f func()) {
	defer func() {
		if recover() == nil {
			panic(name)
		}
	}()
	f()
}
