// errorcheck

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 43673
package main

type I interface {
	M()
}

type T struct{}

func (t *T) M() {}

func main() {
	var i I
	_ = i.(*I) // ERROR "*I does not implement I (but I does)"
}
