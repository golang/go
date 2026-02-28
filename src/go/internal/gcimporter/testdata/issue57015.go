// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue57015

type E error

type X[T any] struct {}

func F() X[interface {
	E
}] {
	panic(0)
}

