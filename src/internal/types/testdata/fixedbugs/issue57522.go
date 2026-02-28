// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// A simplified version of the code in the original report.
type S[T any] struct{}
var V = S[any]{}
func (fs *S[T]) M(V.M /* ERROR "V.M is not a type" */) {}

// Other minimal reproducers.
type S1[T any] V1.M /* ERROR "V1.M is not a type" */
type V1 = S1[any]

type S2[T any] struct{}
type V2 = S2[any]
func (fs *S2[T]) M(x V2.M /* ERROR "V2.M is not a type" */ ) {}

// The following still panics, as the selector is reached from check.expr
// rather than check.typexpr. TODO(rfindley): fix this.
// type X[T any] int
// func (X[T]) M(x [X[int].M]int) {}

