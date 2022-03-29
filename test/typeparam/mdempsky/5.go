// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type X[T any] int

func (X[T]) F(T) {}

func x() {
	X[interface{}](0).F(0)
}
