// errorcheck

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that len non-constants are not constants, https://golang.org/issue/3244.

package p

var b struct {
	a[10]int
}

var m map[string][20]int

var s [][30]int

func f() *[40]int
var c chan *[50]int
var z complex128

const (
	n1 = len(b.a)
	n2 = len(m[""])
	n3 = len(s[10])

	n4 = len(f())  // ERROR "is not a constant|is not constant"
	n5 = len(<-c) // ERROR "is not a constant|is not constant"

	n6 = cap(f())  // ERROR "is not a constant|is not constant"
	n7 = cap(<-c) // ERROR "is not a constant|is not constant"
	n8 = real(z) // ERROR "is not a constant|is not constant"
	n9 = len([4]float64{real(z)}) // ERROR "is not a constant|is not constant"

)

