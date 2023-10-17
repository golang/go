// errorcheck

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Checking that line number is correct in error message.

package main

type Cint int

func foobar(*Cint, Cint, Cint, *Cint)

func main() {
	a := Cint(1)

	foobar(
		&a,
		0,
		0,
		42, // ERROR ".*"
	)
}
