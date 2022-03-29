// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

import "./a"

type (
	lA[P any]               [10]P
	lS[P any]               struct{ f P }
	lP[P any]               *P
	lM[K comparable, V any] map[K]V
)

// local cycles
type (
	A  lA[A]            // ERROR "invalid recursive type"
	S  lS[S]            // ERROR "invalid recursive type"
	P  lP[P]            // ok (indirection through lP)
	M1 lM[int, M1]      // ok (indirection through lM)
	M2 lM[lA[byte], M2] // ok (indirection through lM)

	A2 lA[lS[lP[A2]]] // ok (indirection through lP)
	A3 lA[lS[lS[A3]]] // ERROR "invalid recursive type"
)

// cycles through imported types
type (
	Ai  a.A[Ai]             // ERROR "invalid recursive type"
	Si  a.S[Si]             // ERROR "invalid recursive type"
	Pi  a.P[Pi]             // ok (indirection through a.P)
	M1i a.M[int, M1i]       // ok (indirection through a.M)
	M2i a.M[a.A[byte], M2i] // ok (indirection through a.M)

	A2i a.A[a.S[a.P[A2i]]] // ok (indirection through a.P)
	A3i a.A[a.S[a.S[A3i]]] // ERROR "invalid recursive type"

	T2 a.S[T0[T2]] // ERROR "invalid recursive type"
	T3 T0[Ai]      // no follow-on error here
)

// test case from issue

type T0[P any] struct {
	f P
}

type T1 struct { // ERROR "invalid recursive type"
	_ T0[T1]
}
