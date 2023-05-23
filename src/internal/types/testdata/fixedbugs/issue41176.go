// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type S struct{}

func (S) M() byte {
	return 0
}

type I[T any] interface {
	M() T
}

func f[T any](x I[T]) {}

func _() {
	f(S{})
}
