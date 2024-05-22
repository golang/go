// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check correctness of various closure corner cases
// that are expected to be inlined

package main

var ok bool
var sink int

func main() {
	{
		if x := func() int { // ERROR "can inline main.func1"
			return 1
		}(); x != 1 { // ERROR "inlining call to main.func1"
			ppanic("x != 1")
		}
		if x := func() int { // ERROR "can inline main.func2" "func literal does not escape"
			return 1
		}; x() != 1 { // ERROR "inlining call to main.func2"
			_ = x // prevent simple deadcode elimination after inlining
			ppanic("x() != 1")
		}
	}

	{
		if y := func(x int) int { // ERROR "can inline main.func3"
			return x + 2
		}(40); y != 42 { // ERROR "inlining call to main.func3"
			ppanic("y != 42")
		}
		if y := func(x int) int { // ERROR "can inline main.func4" "func literal does not escape"
			return x + 2
		}; y(40) != 42 { // ERROR "inlining call to main.func4"
			_ = y // prevent simple deadcode elimination after inlining
			ppanic("y(40) != 42")
		}
	}

	{
		y := func(x int) int { // ERROR "can inline main.func5" "func literal does not escape"
			return x + 2
		}
		y = func(x int) int { // ERROR "can inline main.func6" "func literal does not escape"
			return x + 1
		}
		if y(40) != 41 {
			ppanic("y(40) != 41")
		}
	}

	{
		func() { // ERROR "func literal does not escape"
			y := func(x int) int { // ERROR "can inline main.func7.1" "func literal does not escape"
				return x + 2
			}
			y = func(x int) int { // ERROR "can inline main.func7.2" "func literal does not escape"
				return x + 1
			}
			if y(40) != 41 {
				ppanic("y(40) != 41")
			}
		}()
	}

	{
		y := func(x int) int { // ERROR "can inline main.func8" "func literal does not escape"
			return x + 2
		}
		y, sink = func(x int) int { // ERROR "can inline main.func9" "func literal does not escape"
			return x + 1
		}, 42
		if y(40) != 41 {
			ppanic("y(40) != 41")
		}
	}

	{
		func() { // ERROR "func literal does not escape"
			y := func(x int) int { // ERROR "can inline main.func10.1" "func literal does not escape"
				return x + 2
			}
			y, sink = func(x int) int { // ERROR "can inline main.func10.2" "func literal does not escape"
				return x + 1
			}, 42
			if y(40) != 41 {
				ppanic("y(40) != 41")
			}
		}()
	}

	{
		y := func(x int) int { // ERROR "can inline main.func11" "func literal does not escape"
			return x + 2
		}
		y, sink = func() (func(int) int, int) { // ERROR "can inline main.func12"
			return func(x int) int { // ERROR "can inline main.func12" "func literal escapes to heap"
				return x + 1
			}, 42
		}() // ERROR "func literal does not escape" "inlining call to main.func12"
		if y(40) != 41 {
			ppanic("y(40) != 41")
		}
	}

	{
		func() { // ERROR "func literal does not escape"
			y := func(x int) int { // ERROR "func literal does not escape" "can inline main.func13.1"
				return x + 2
			}
			y, sink = func() (func(int) int, int) { // ERROR "can inline main.func13.2"
				return func(x int) int { // ERROR   "can inline main.func13.2" "func literal escapes to heap"
					return x + 1
				}, 42
			}() // ERROR "func literal does not escape" "inlining call to main.func13.2"
			if y(40) != 41 {
				ppanic("y(40) != 41")
			}
		}()
	}

	{
		y := func(x int) int { // ERROR "can inline main.func14" "func literal does not escape"
			return x + 2
		}
		y, ok = map[int]func(int) int{ // ERROR "does not escape"
			0: func(x int) int { return x + 1 }, // ERROR "can inline main.func15" "func literal escapes"
		}[0]
		if y(40) != 41 {
			ppanic("y(40) != 41")
		}
	}

	{
		func() { // ERROR "func literal does not escape"
			y := func(x int) int { // ERROR "can inline main.func16.1" "func literal does not escape"
				return x + 2
			}
			y, ok = map[int]func(int) int{ // ERROR "does not escape"
				0: func(x int) int { return x + 1 }, // ERROR "can inline main.func16.2" "func literal escapes"
			}[0]
			if y(40) != 41 {
				ppanic("y(40) != 41")
			}
		}()
	}

	{
		y := func(x int) int { // ERROR "can inline main.func17" "func literal does not escape"
			return x + 2
		}
		y, ok = interface{}(func(x int) int { // ERROR "can inline main.func18" "does not escape"
			return x + 1
		}).(func(int) int)
		if y(40) != 41 {
			ppanic("y(40) != 41")
		}
	}

	{
		func() { // ERROR "func literal does not escape"
			y := func(x int) int { // ERROR "can inline main.func19.1" "func literal does not escape"
				return x + 2
			}
			y, ok = interface{}(func(x int) int { // ERROR "can inline main.func19.2" "does not escape"
				return x + 1
			}).(func(int) int)
			if y(40) != 41 {
				ppanic("y(40) != 41")
			}
		}()
	}

	{
		x := 42
		if y := func() int { // ERROR "can inline main.func20"
			return x
		}(); y != 42 { // ERROR "inlining call to main.func20"
			ppanic("y != 42")
		}
		if y := func() int { // ERROR "can inline main.func21" "func literal does not escape"
			return x
		}; y() != 42 { // ERROR "inlining call to main.func21"
			_ = y // prevent simple deadcode elimination after inlining
			ppanic("y() != 42")
		}
	}

	{
		x := 42
		if z := func(y int) int { // ERROR "can inline main.func22"
			return func() int { // ERROR "can inline main.func22.1" "can inline main.main.func22.func30"
				return x + y
			}() // ERROR "inlining call to main.func22.1"
		}(1); z != 43 { // ERROR "inlining call to main.func22" "inlining call to main.main.func22.func30"
			ppanic("z != 43")
		}
		if z := func(y int) int { // ERROR "func literal does not escape" "can inline main.func23"
			return func() int { // ERROR "can inline main.func23.1" "can inline main.main.func23.func31"
				return x + y
			}() // ERROR "inlining call to main.func23.1"
		}; z(1) != 43 { // ERROR "inlining call to main.func23" "inlining call to main.main.func23.func31"
			_ = z // prevent simple deadcode elimination after inlining
			ppanic("z(1) != 43")
		}
	}

	{
		a := 1
		func() { // ERROR "can inline main.func24"
			func() { // ERROR "can inline main.func24" "can inline main.main.func24.func32"
				a = 2
			}() // ERROR "inlining call to main.func24"
		}() // ERROR "inlining call to main.func24" "inlining call to main.main.func24.func32"
		if a != 2 {
			ppanic("a != 2")
		}
	}

	{
		b := 2
		func(b int) { // ERROR "func literal does not escape"
			func() { // ERROR "can inline main.func25.1"
				b = 3
			}() // ERROR "inlining call to main.func25.1"
			if b != 3 {
				ppanic("b != 3")
			}
		}(b)
		if b != 2 {
			ppanic("b != 2")
		}
	}

	{
		c := 3
		func() { // ERROR "can inline main.func26"
			c = 4
			func() {
				if c != 4 {
					ppanic("c != 4")
				}
				recover() // prevent inlining
			}()
		}() // ERROR "inlining call to main.func26" "func literal does not escape"
		if c != 4 {
			ppanic("c != 4")
		}
	}

	{
		a := 2
		// This has an unfortunate exponential growth, where as we visit each
		// function, we inline the inner closure, and that constructs a new
		// function for any closures inside the inner function, and then we
		// revisit those. E.g., func34 and func36 are constructed by the inliner.
		if r := func(x int) int { // ERROR "can inline main.func27"
			b := 3
			return func(y int) int { // ERROR "can inline main.func27.1" "can inline main.main.func27.func34"
				c := 5
				return func(z int) int { // ERROR "can inline main.func27.1.1" "can inline main.main.func27.func34.1" "can inline main.func27.main.func27.1.2" "can inline main.main.func27.main.main.func27.func34.func36"
					return a*x + b*y + c*z
				}(10) // ERROR "inlining call to main.func27.1.1"
			}(100) // ERROR "inlining call to main.func27.1" "inlining call to main.func27.main.func27.1.2"
		}(1000); r != 2350 { // ERROR "inlining call to main.func27" "inlining call to main.main.func27.func34" "inlining call to main.main.func27.main.main.func27.func34.func36"
			ppanic("r != 2350")
		}
	}

	{
		a := 2
		if r := func(x int) int { // ERROR "can inline main.func28"
			b := 3
			return func(y int) int { // ERROR "can inline main.func28.1" "can inline main.main.func28.func35"
				c := 5
				func(z int) { // ERROR "can inline main.func28.1.1" "can inline main.func28.main.func28.1.2" "can inline main.main.func28.func35.1" "can inline main.main.func28.main.main.func28.func35.func37"
					a = a * x
					b = b * y
					c = c * z
				}(10) // ERROR "inlining call to main.func28.1.1"
				return a + c
			}(100) + b // ERROR "inlining call to main.func28.1" "inlining call to main.func28.main.func28.1.2"
		}(1000); r != 2350 { // ERROR "inlining call to main.func28" "inlining call to main.main.func28.func35" "inlining call to main.main.func28.main.main.func28.func35.func37"
			ppanic("r != 2350")
		}
		if a != 2000 {
			ppanic("a != 2000")
		}
	}
}

//go:noinline
func notmain() {
	{
		// This duplicates the first block in main, but without the "_ = x" for closure x.
		// This allows dead code elimination of x before escape analysis,
		// thus "func literal does not escape" should not appear.
		if x := func() int { // ERROR "can inline notmain.func1"
			return 1
		}(); x != 1 { // ERROR "inlining call to notmain.func1"
			ppanic("x != 1")
		}
		if x := func() int { // ERROR "can inline notmain.func2"
			return 1
		}; x() != 1 { // ERROR "inlining call to notmain.func2"
			ppanic("x() != 1")
		}
	}
}

//go:noinline
func ppanic(s string) { // ERROR "leaking param: s"
	panic(s) // ERROR "s escapes to heap"
}
