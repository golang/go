// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check that these do not use "by value" capturing,
// because changes are made to the value during the closure.

package main

var never bool

func main() {
	{
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
	}

	{
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
	}

	{
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

	{
		var g func() int
		for i := range [2]int{} {
			if i == 0 {
				g = func() int {
					return i // test that we capture by ref here, i is mutated on every interaction
				}
			}
		}
		if g() != 1 {
			panic("g() != 1")
		}
	}

	{
		var g func() int
		q := 0
		for range [2]int{} {
			q++
			g = func() int {
				return q // test that we capture by ref here
				// q++ must on a different decldepth than q declaration
			}
		}
		if g() != 2 {
			panic("g() != 2")
		}
	}

	{
		var g func() int
		var a [2]int
		q := 0
		for a[func() int {
			q++
			return 0
		}()] = range [2]int{} {
			g = func() int {
				return q // test that we capture by ref here
				// q++ must on a different decldepth than q declaration
			}
		}
		if g() != 2 {
			panic("g() != 2")
		}
	}

	{
		var g func() int
		q := 0
		q, g = 1, func() int { return q }
		if never {
			g = func() int { return 2 }
		}
		if g() != 1 {
			panic("g() != 1")
		}
	}
}
