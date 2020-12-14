// errorcheck

// Copyright 2020 The Go Authors. All rights reserved.  Use of this
// source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package p

type T1 struct { // ERROR "invalid recursive type T1\n\tLINE: T1 refers to\n\tLINE+4: T2 refers to\n\tLINE: T1$"
	f2 T2
}

type T2 struct {
	f1 T1
}

type a b
type b c // ERROR "invalid recursive type b\n\tLINE: b refers to\n\tLINE+1: c refers to\n\tLINE: b$"
type c b

type d e
type e f
type f f // ERROR "invalid recursive type f\n\tLINE: f refers to\n\tLINE: f$"

type g struct { // ERROR "invalid recursive type g\n\tLINE: g refers to\n\tLINE: g$"
	h struct {
		g
	}
}

type w x
type x y // ERROR "invalid recursive type x\n\tLINE: x refers to\n\tLINE+1: y refers to\n\tLINE+2: z refers to\n\tLINE: x$"
type y struct{ z }
type z [10]x

type w2 w // refer to the type loop again
