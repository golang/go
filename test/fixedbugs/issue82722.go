// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:noinline
func run(f func()) { f() }

func F1() func() {
	x := 1
	return func() {
		check := func() {
			run(func() {
				go func() {
					if x != 1 {
						panic("unexpected x value")
					}
				}()
			})
		}
		check()
	}
}

func F2() func() {
	x := 42
	return func() {
		x := x
		y := x
		z := 42
		check := func() {
			run(func() {
				go func() {
					if x != 42 {
						panic("unexpected x value")
					}
					if y != 42 || y != x {
						panic("unexpected y value")
					}
					if z != 42 {
						panic("unexpected z value")
					}
				}()
			})
		}
		check()
	}
}

//go:noinline
func use(a int64) {}

func G() {
	x := int64(1000)
	run(func() {
		g := func() {
			run(func() {
				func() {
					use(x)
					if x != 1000 {
						panic("unexpected x value")
					}
				}()
			})
		}
		g()
	})
}

func main() {
	F1()()
	F2()()
	G()
}
