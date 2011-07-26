// errchk $G -e $D/$F.go

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "unsafe"

func sum(args ...int) int { return 0 }

var (
	_ = sum(1, 2, 3)
	_ = sum()
	_ = sum(1.0, 2.0)
	_ = sum(1.5)      // ERROR "integer"
	_ = sum("hello")  // ERROR "string.*as type int|incompatible"
	_ = sum([]int{1}) // ERROR "slice literal.*as type int|incompatible"
)

type T []T

func funny(args ...T) int { return 0 }

var (
	_ = funny(nil)
	_ = funny(nil, nil)
	_ = funny([]T{}) // ok because []T{} is a T; passes []T{[]T{}}
)

func bad(args ...int) {
	print(1, 2, args...)	// ERROR "[.][.][.]"
	println(args...)	// ERROR "[.][.][.]"
	ch := make(chan int)
	close(ch...)	// ERROR "[.][.][.]"
	_ = len(args...)	// ERROR "[.][.][.]"
	_ = new(int...)	// ERROR "[.][.][.]"
	n := 10
	_ = make([]byte, n...)	// ERROR "[.][.][.]"
	// TODO(rsc): enable after gofmt bug is fixed
	//	_ = make([]byte, 10 ...)	// error "[.][.][.]"
	var x int
	_ = unsafe.Pointer(&x...)	// ERROR "[.][.][.]"
	_ = unsafe.Sizeof(x...)	// ERROR "[.][.][.]"
	_ = [...]byte("foo") // ERROR "[.][.][.]"
	_ = [...][...]int{{1,2,3},{4,5,6}}	// ERROR "[.][.][.]"
}

