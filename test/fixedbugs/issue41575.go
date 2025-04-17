// errorcheck

// Copyright 2020 The Go Authors. All rights reserved.  Use of this
// source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package p

type T1 struct { // ERROR "invalid recursive type T1\n.*T1 refers to T2\n.*T2 refers to T1|invalid recursive type"
	f2 T2
}

type T2 struct { // GCCGO_ERROR "invalid recursive type"
	f1 T1
}

type a b // GCCGO_ERROR "invalid recursive type"
type b c // ERROR "invalid recursive type b\n.*b refers to c\n.*c refers to b|invalid recursive type|invalid recursive type"
type c b // GCCGO_ERROR "invalid recursive type"

type d e
type e f
type f f // ERROR "invalid recursive type: f refers to itself|invalid recursive type|invalid recursive type"

type g struct { // ERROR "invalid recursive type: g refers to itself|invalid recursive type"
	h struct {
		g
	}
}

type w x
type x y           // ERROR "invalid recursive type x\n.*x refers to y\n.*y refers to z\n.*z refers to x|invalid recursive type"
type y struct{ z } // GCCGO_ERROR "invalid recursive type"
type z [10]x

type w2 w // refer to the type loop again
