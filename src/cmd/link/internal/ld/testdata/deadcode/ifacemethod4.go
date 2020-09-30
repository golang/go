// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that a live type's method is not live even if
// it matches an interface method, as long as the interface
// method is not used.

package main

type T int

func (T) M() {}

type I interface{ M() }

var p *T
var pp *I

func main() {
	p = new(T)  // use type T
	pp = new(I) // use type I
}
