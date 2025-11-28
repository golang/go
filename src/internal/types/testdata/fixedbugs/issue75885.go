// -gotypesalias=1

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type A[P any] = P // ERROR "cannot use type parameter declared in alias declaration as RHS"

func _[P any]() {
	type A[P any] = P // ERROR "cannot use type parameter declared in alias declaration as RHS"
	type B = P
	type C[Q any] = P
}
