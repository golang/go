// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type Option[T any] interface {
	ToSeq() Seq[T]
}

type Seq[T any] []T

func (r Seq[T]) Find(p func(v T) bool) Option[T] {
	panic("")
}
