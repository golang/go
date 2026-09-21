// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const targetPC = uintptr(0xdeadbeef)

type payload struct {
	x uintptr
	y *uintptr
	fn   [2]func()
}

var p payload
var v []byte

func init() {
	p.x = targetPC
	p.y = &p.x
	p.fn[0] = func() {}
	p.fn[1] = func() {}
}

//go:noinline
func trigger(n int) {
	defer func() { recover() }()

	if n < len(p.fn) {
		p.fn[n&1]()

		s := make([]byte, n)
		v = s
	}
}

func main() {
	trigger(-1)
}
