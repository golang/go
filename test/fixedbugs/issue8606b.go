// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is an optimization check. We want to make sure that we compare
// string lengths, and other scalar fields, before checking string
// contents.  There's no way to verify this in the language, and
// codegen tests in test/codegen can't really detect ordering
// optimizations like this. Instead, we generate invalid strings with
// bad backing store pointers but nonzero length, so we can check that
// the backing store never gets compared.
//
// We use two different bad strings so that pointer comparisons of
// backing store pointers fail.

package main

import (
	"fmt"
	"reflect"
	"unsafe"
)

func bad1() string {
	s := "foo"
	(*reflect.StringHeader)(unsafe.Pointer(&s)).Data = 1 // write bad value to data ptr
	return s
}
func bad2() string {
	s := "foo"
	(*reflect.StringHeader)(unsafe.Pointer(&s)).Data = 2 // write bad value to data ptr
	return s
}

type SI struct {
	s string
	i int
}

type SS struct {
	s string
	t string
}

func main() {
	for _, test := range []struct {
		a, b interface{}
	}{
		{SI{s: bad1(), i: 1}, SI{s: bad2(), i: 2}},
		{SS{s: bad1(), t: "a"}, SS{s: bad2(), t: "aa"}},
		{SS{s: "a", t: bad1()}, SS{s: "b", t: bad2()}},
		// This one would panic because the length of both strings match, and we check
		// the body of the bad strings before the body of the good strings.
		//{SS{s: bad1(), t: "a"}, SS{s: bad2(), t: "b"}},
	} {
		if test.a == test.b {
			panic(fmt.Sprintf("values %#v and %#v should not be equal", test.a, test.b))
		}
	}

}
