// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// The example from the issue.
type (
	N[P any] M /* ERROR "invalid recursive type" */ [P]
	M[P any] N[P]
)

// A slightly more complicated case.
type (
	A[P any] B /* ERROR "invalid recursive type" */ [P]
	B[P any] C[P]
	C[P any] A[P]
)

// Confusing but valid (note that `type T *T` is valid).
type (
	N1[P any] *M1[P]
	M1[P any] *N1[P]
)
