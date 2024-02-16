// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T[_ any] struct {}

func (T /* ERROR "instantiation" */ ) m()

func _() {
	var x interface { m() }
	x = T[int]{}
	_ = x
}
