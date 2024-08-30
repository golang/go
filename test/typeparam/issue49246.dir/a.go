// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type R[T any] struct{ v T }

func (r R[T]) Self() R[T] { return R[T]{} }

type Fn[T any] func() R[T]

func X() (r R[int]) { return r.Self() }

func Y[T any](a Fn[T]) Fn[int] {
	return func() (r R[int]) {
		// No crash: return R[int]{}
		return r.Self()
	}
}
