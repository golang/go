// errchk $G $D/$F.go

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 2231

package main
import "runtime"

func foo(runtime.UintType, i int) {  // ERROR "cannot declare name runtime.UintType|named/anonymous mix"
	println(i, runtime.UintType) 
}

func bar(i int) {
	runtime.UintType := i       // ERROR "cannot declare name runtime.UintType|non-name on left side"
	println(runtime.UintType)	// GCCGO_ERROR "invalid use of type"
}

func baz() {
	main.i := 1	// ERROR "non-name main.i|non-name on left side"
	println(main.i)	// GCCGO_ERROR "no fields or methods"
}

func qux() {
	var main.i	// ERROR "unexpected [.]|expected type"
	println(main.i)
}

func corge() {
	var foo.i int  // ERROR "unexpected [.]|expected type"
	println(foo.i)
}

func main() {
	foo(42,43)
	bar(1969)
}
