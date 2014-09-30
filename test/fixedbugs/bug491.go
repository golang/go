// run

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test order of calls to builtin functions.
// Discovered during CL 144530045 review.

package main

func main() {
	// append
	{
		x := make([]int, 0)
		f := func() int { x = make([]int, 2); return 2 }
		a, b, c := append(x, 1), f(), append(x, 1)
		if len(a) != 1 || len(c) != 3 {
			bug()
			println("append call not ordered:", len(a), b, len(c))
		}
	}

	// cap
	{
		x := make([]int, 1)
		f := func() int { x = make([]int, 3); return 2 }
		a, b, c := cap(x), f(), cap(x)
		if a != 1 || c != 3 {
			bug()
			println("cap call not ordered:", a, b, c)
		}
	}

	// complex
	{
		x := 1.0
		f := func() int { x = 3; return 2 }
		a, b, c := complex(x, 0), f(), complex(x, 0)
		if real(a) != 1 || real(c) != 3 {
			bug()
			println("complex call not ordered:", a, b, c)
		}
	}

	// copy
	{
		tmp := make([]int, 100)
		x := make([]int, 1)
		f := func() int { x = make([]int, 3); return 2 }
		a, b, c := copy(tmp, x), f(), copy(tmp, x)
		if a != 1 || c != 3 {
			bug()
			println("copy call not ordered:", a, b, c)
		}
	}

	// imag
	{
		x := 1i
		f := func() int { x = 3i; return 2 }
		a, b, c := imag(x), f(), imag(x)
		if a != 1 || c != 3 {
			bug()
			println("imag call not ordered:", a, b, c)
		}
	}

	// len
	{
		x := make([]int, 1)
		f := func() int { x = make([]int, 3); return 2 }
		a, b, c := len(x), f(), len(x)
		if a != 1 || c != 3 {
			bug()
			println("len call not ordered:", a, b, c)
		}
	}

	// make
	{
		x := 1
		f := func() int { x = 3; return 2 }
		a, b, c := make([]int, x), f(), make([]int, x)
		if len(a) != 1 || len(c) != 3 {
			bug()
			println("make call not ordered:", len(a), b, len(c))
		}
	}

	// real
	{
		x := 1 + 0i
		f := func() int { x = 3; return 2 }
		a, b, c := real(x), f(), real(x)
		if a != 1 || c != 3 {
			bug()
			println("real call not ordered:", a, b, c)
		}
	}
}

var bugged = false

func bug() {
	if !bugged {
		println("BUG")
		bugged = true
	}
}