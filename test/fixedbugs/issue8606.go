// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check to make sure that we compare fields in order. See issue 8606.

package main

import "fmt"

func main() {
	type A [2]interface{}
	type S struct{ x, y interface{} }

	for _, test := range []struct {
		panic bool
		a, b  interface{}
	}{
		{false, A{1, []byte{1}}, A{2, []byte{1}}},
		{true, A{[]byte{1}, 1}, A{[]byte{1}, 2}},
		{false, S{1, []byte{1}}, S{2, []byte{1}}},
		{true, S{[]byte{1}, 1}, S{[]byte{1}, 2}},
	} {
		f := func() {
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
