// errorcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T[P any] struct {
	_ P
}

type A T[A] // ERROR "invalid recursive type A\n.*A refers to T\[A\]\n.*T\[A\] refers to A"

type B = C
type C T[B] // ERROR "invalid recursive type C\n.*C refers to T\[B\]\n.*T\[B\] refers to C"

type D = T[D] // ERROR "invalid recursive type: D refers to itself"

type E T[T[E]] // ERROR "invalid recursive type E\n.*E refers to T\[T\[E\]\]\n.*T\[T\[E\]\] refers to T\[E\]\n.*T\[E\] refers to E"
