// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check correctness of various closure corner cases that
// that are expected to be inlined

package main

var ok bool
var sink int

func main() {
	{
		if x := func() int { // ERROR "can inline main.func1"
			return 1
		}(); x != 1 { // ERROR "inlining call to main.func1"
			panic("x != 1")
		}
		if x := func() int { // ERROR "can inline main.func2" "func literal does not escape"
			return 1
		}; x() != 1 { // ERROR "inlining call to main.func2"
			panic("x() != 1")
		}
	}

	{
		if y := func(x int) int { // ERROR "can inline main.func3"
			return x + 2
		}(40); y != 42 { // ERROR "inlining call to main.func3"
			panic("y != 42")
		}
		if y := func(x int) int { // ERROR "can inline main.func4" "func literal does not escape"
			return x + 2
		}; y(40) != 42 { // ERROR "inlining call to main.func4"
			panic("y(40) != 42")
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
			panic("y(40) != 41")
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
				panic("y(40) != 41")
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
			panic("y(40) != 41")
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
				panic("y(40) != 41")
			}
		}()
	}

	{
		y := func(x int) int { // ERROR "can inline main.func11" "func literal does not escape"
			return x + 2
		}
		y, sink = func() (func(int)int, int) { // ERROR "func literal does not escape"
			return func(x int) int { // ERROR "can inline main.func12" "func literal escapes"
				return x + 1
			}, 42
		}()
		if y(40) != 41 {
			panic("y(40) != 41")
		}
	}

	{
		func() { // ERROR "func literal does not escape"
			y := func(x int) int { // ERROR "can inline main.func13.1" "func literal does not escape"
				return x + 2
			}
			y, sink = func() (func(int) int, int) { // ERROR "func literal does not escape"
				return func(x int) int { // ERROR "can inline main.func13.2" "func literal escapes"
					return x + 1
				}, 42
			}()
			if y(40) != 41 {
				panic("y(40) != 41")
			}
		}()
	}

	{
		y := func(x int) int { // ERROR "can inline main.func14" "func literal does not escape"
			return x + 2
		}
		y, ok = map[int]func(int)int { // ERROR "does not escape"
			0: func (x int) int { return x + 1 }, // ERROR "can inline main.func15" "func literal escapes"
		}[0]
		if y(40) != 41 {
			panic("y(40) != 41")
		}
	}

	{
		func() { // ERROR "func literal does not escape"
			y := func(x int) int { // ERROR "can inline main.func16.1" "func literal does not escape"
				return x + 2
			}
			y, ok = map[int]func(int) int{// ERROR "does not escape"
				0: func(x int) int { return x + 1 }, // ERROR "can inline main.func16.2" "func literal escapes"
			}[0]
			if y(40) != 41 {
				panic("y(40) != 41")
			}
		}()
	}

	{
		y := func(x int) int { // ERROR "can inline main.func17" "func literal does not escape"
			return x + 2
		}
		y, ok = interface{}(func (x int) int { // ERROR "can inline main.func18" "does not escape"
			return x + 1
		}).(func(int)int)
		if y(40) != 41 {
			panic("y(40) != 41")
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
				panic("y(40) != 41")
			}
		}()
	}
}
