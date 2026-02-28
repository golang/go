// errorcheck

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that illegal uses of ... are detected.
// Does not compile.

package main

import "unsafe"

func sum(args ...int) int { return 0 }

var (
	_ = sum(1, 2, 3)
	_ = sum()
	_ = sum(1.0, 2.0)
	_ = sum(1.5)      // ERROR "1\.5 .untyped float constant. as int|integer"
	_ = sum("hello")  // ERROR ".hello. (.untyped string constant. as int|.type untyped string. as type int)|incompatible"
	_ = sum([]int{1}) // ERROR "\[\]int{.*}.*as type int"
)

func sum3(int, int, int) int { return 0 }
func tuple() (int, int, int) { return 1, 2, 3 }

var (
	_ = sum(tuple())
	_ = sum(tuple()...) // ERROR "\.{3} with 3-valued|multiple-value"
	_ = sum3(tuple())
	_ = sum3(tuple()...) // ERROR "\.{3} in call to non-variadic|multiple-value|invalid use of .*[.][.][.]"
)

type T []T

func funny(args ...T) int { return 0 }

var (
	_ = funny(nil)
	_ = funny(nil, nil)
	_ = funny([]T{}) // ok because []T{} is a T; passes []T{[]T{}}
)

func Foo(n int) {}

func bad(args ...int) {
	print(1, 2, args...)	// ERROR "[.][.][.]"
	println(args...)	// ERROR "[.][.][.]"
	ch := make(chan int)
	close(ch...)	// ERROR "[.][.][.]"
	_ = len(args...)	// ERROR "[.][.][.]"
	_ = new(int...)	// ERROR "[.][.][.]"
	n := 10
	_ = make([]byte, n...)	// ERROR "[.][.][.]"
	_ = make([]byte, 10 ...)	// ERROR "[.][.][.]"
	var x int
	_ = unsafe.Pointer(&x...)	// ERROR "[.][.][.]"
	_ = unsafe.Sizeof(x...)	// ERROR "[.][.][.]"
	_ = [...]byte("foo") // ERROR "[.][.][.]"
	_ = [...][...]int{{1,2,3},{4,5,6}}	// ERROR "[.][.][.]"

	Foo(x...) // ERROR "\.{3} in call to non-variadic|invalid use of .*[.][.][.]"
}
