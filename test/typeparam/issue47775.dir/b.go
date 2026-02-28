// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

type C[T any] struct {
}

func (c *C[T]) reset() {
}

func New[T any]() {
	c := &C[T]{}
	z(c.reset)
}

func z(interface{}) {
}
