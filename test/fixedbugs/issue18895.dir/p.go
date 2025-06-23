// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func F() { // ERROR "can inline"
	var v t
	v.m() // ERROR "inlining call"
}

type t int

func (t) m() {} // ERROR "can inline"
