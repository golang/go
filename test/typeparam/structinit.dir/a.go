// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type S[T any] struct {
}

func (b *S[T]) build() *X[T] {
	return &X[T]{f:0}
}
type X[T any] struct {
	f int
}
