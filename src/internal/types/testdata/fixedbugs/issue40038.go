// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type A[T any] int

func (A[T]) m(A[T])

func f[P interface{m(P)}]() {}

func _() {
	_ = f[A[int]]
}
