// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T int

func (T) m() { println("m") }

type I interface{ m() }

func F() {
	i.m()
}

var i I = T(123)
