// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check that these do not use "by value" capturing,
// because changes are made to the value during the closure.

package main

func main() {
	type X struct {
		v int
	}
	var x X
	func() {
		x.v++
	}()
	if x.v != 1 {
		panic("x.v != 1")
	}

	type Y struct {
		X
	}
	var y Y
	func() {
		y.v = 1
	}()
	if y.v != 1 {
		panic("y.v != 1")
	}

	type Z struct {
		a [3]byte
	}
	var z Z
	func() {
		i := 0
		for z.a[1] = 1; i < 10; i++ {
		}
	}()
	if z.a[1] != 1 {
		panic("z.a[1] != 1")
	}

	w := 0
	tmp := 0
	f := func() {
		if w != 1 {
			panic("w != 1")
		}
	}
	func() {
		tmp = w // force capture of w, but do not write to it yet
		_ = tmp
		func() {
			func() {
				w++ // write in a nested closure
			}()
		}()
	}()
	f()
}
