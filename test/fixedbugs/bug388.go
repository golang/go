// errorcheck

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 2231

package main
import "runtime"

func foo(runtime.UintType, i int) {  // ERROR "cannot declare name runtime.UintType|mixed named and unnamed|undefined identifier"
	println(i, runtime.UintType) // GCCGO_ERROR "undefined identifier"
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
