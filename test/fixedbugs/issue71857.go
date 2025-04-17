// run

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "sync/atomic"

//go:noinline
func f(p0, p1, p2, p3, p4, p5, p6, p7 *uint64, a *atomic.Uint64) {
	old := a.Or(0xaaa)
	*p0 = old
	*p1 = old
	*p2 = old
	*p3 = old
	*p4 = old
	*p5 = old
	*p6 = old
	*p7 = old
}

func main() {
	a := new(atomic.Uint64)
	p := new(uint64)
	f(p, p, p, p, p, p, p, p, a)

}
