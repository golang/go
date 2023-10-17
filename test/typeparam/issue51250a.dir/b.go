// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

import "./a"

type T struct { a int }

var I interface{} = a.G[T]{}

//go:noinline
func F(x interface{}) {
	switch x.(type) {
	case a.G[T]:
	case int:
		panic("bad")
	case float64:
		panic("bad")
	default:
		panic("bad")
	}
}
