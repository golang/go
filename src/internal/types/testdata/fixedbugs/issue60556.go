// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type I[T any] interface {
	m(I[T])
}

type S[T any] struct{}

func (S[T]) m(I[T]) {}

func f[T I[E], E any](T) {}

func _() {
	f(S[int]{})
}
