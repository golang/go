// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// Example from the issue:
type A[T any] interface {
	*T
}

type B[T any] interface {
	B /* ERROR "invalid recursive type" */ [*T]
}

type C[T any, U B[U]] interface {
	*T
}

// Simplified reproducer:
type X[T any] interface {
	X /* ERROR "invalid recursive type" */ [*T]
}

var _ X[int]

// A related example that doesn't go through interfaces.
type A2[P any] [10]A2 /* ERROR "invalid recursive type" */ [*P]

var _ A2[int]
