// errchk $G -e $D/$F.go

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func use(bool) {}

type T1 *int
type T2 *int

type T3 struct {}

var t3 T3

func main() {
	// Arguments to comparison must be
	// assignable one to the other (or vice versa)
	// so chan int can be compared against
	// directional channels but channel of different
	// direction cannot be compared against each other.
	var c1 chan <-int
	var c2 <-chan int
	var c3 chan int
	
	use(c1 == c2)	// ERROR "invalid operation|incompatible"
	use(c2 == c1)	// ERROR "invalid operation|incompatible"
	use(c1 == c3)
	use(c2 == c2)
	use(c3 == c1)
	use(c3 == c2)

	// Same applies to named types.
	var p1 T1
	var p2 T2
	var p3 *int
	
	use(p1 == p2)	// ERROR "invalid operation|incompatible"
	use(p2 == p1)	// ERROR "invalid operation|incompatible"
	use(p1 == p3)
	use(p2 == p2)
	use(p3 == p1)
	use(p3 == p2)
	
	// Comparison of structs should have a good message
	use(t3 == t3)	// ERROR "struct|expected"
}
