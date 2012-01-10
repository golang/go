// errchk $G $D/$F.go

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 2231

package main
import "runtime"

func foo(runtime.UintType, i int) {  // ERROR "cannot declare name runtime.UintType"
	println(i, runtime.UintType) 
}

func bar(i int) {
	runtime.UintType := i       // ERROR "cannot declare name runtime.UintType"
	println(runtime.UintType)
}

func baz() {
	main.i := 1	// ERROR "non-name main.i"
	println(main.i)
}

func qux() {
	var main.i	// ERROR "unexpected [.]"
	println(main.i)
}

func corge() {
	var foo.i int  // ERROR "unexpected [.]"
	println(foo.i)
}

func main() {
	foo(42,43)
	bar(1969)
}
