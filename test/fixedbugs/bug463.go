// errorcheck

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 3757: unhelpful typechecking loop message
// for constants that refer to themselves.

package main

const a = a // ERROR "refers to itself|definition loop|initialization loop"

const (
	X    = A
	A    = B // ERROR "refers to itself|definition loop|initialization loop"
	B    = D
	C, D = 1, A
)

func main() {
}
