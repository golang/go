// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type Index[T any] interface {
	G() T
}

type I1[T any] struct {
	a T
}

func (i *I1[T]) G() T {
	return i.a
}
