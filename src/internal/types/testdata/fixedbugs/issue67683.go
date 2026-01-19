// -gotypesalias=1

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type A[P any] func()

// alias signature types
type B[P any] = func()
type C[P any] = B[P]

var _ = A /* ERROR "cannot use generic type A without instantiation" */ (nil)

// generic alias signature types must be instantiated before use
var _ = B /* ERROR "cannot use generic type B without instantiation" */ (nil)
var _ = C /* ERROR "cannot use generic type C without instantiation" */ (nil)
