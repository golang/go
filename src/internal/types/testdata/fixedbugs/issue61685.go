// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _[T any](x any) {
	f /* ERROR "T (type I[T]) does not satisfy I[T] (wrong type for method m)" */ (x.(I[T]))
}

func f[T I[T]](T) {}

type I[T any] interface {
	m(T)
}
