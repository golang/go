// compile

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type S struct {
	a [1 << 16]byte
}

func f1() {
	p := &S{}
	_ = p
}

type T [1 << 16]byte

func f2() {
	p := &T{}
	_ = p
}
