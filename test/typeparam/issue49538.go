// compile -G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type I interface {
	M(interface{})
}

type a[T any] struct{}

func (a[T]) M(interface{}) {}

func f[T I](t *T) {
	(*t).M(t)
}

func g() {
	f(&a[int]{})
}
