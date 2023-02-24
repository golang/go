// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// parameterized types with self-recursive constraints
type (
	T1 /* ERROR "invalid recursive type" */ [P T1[P]]                            interface{}
	T2 /* ERROR "invalid recursive type" */ [P, Q T2[P, Q]]                      interface{}
	T3[P T2[P, Q], Q interface{ ~string }] interface{}

	T4a /* ERROR "invalid recursive type" */ [P T4a[P]]                                                        interface{ ~int }
	T4b /* ERROR "invalid recursive type" */ [P T4b[int]]                                                      interface{ ~int }
	T4c /* ERROR "invalid recursive type" */ [P T4c[string]] interface{ ~int }

	// mutually recursive constraints
	T5 /* ERROR "invalid recursive type" */ [P T6[P]] interface{ int }
	T6[P T5[P]] interface{ int }
)

// verify that constraints are checked as expected
var (
	_ T1[int]
	_ T2[int, string]
	_ T3[int, string]
)

// test case from issue

type Eq /* ERROR "invalid recursive type" */ [a Eq[a]] interface {
	Equal(that a) bool
}
