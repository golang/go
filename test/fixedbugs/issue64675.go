// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func cleanup() {}

//go:noinline
func twoContexts(value int) (int, int, int) {
	count := value
	fn := func(delta int) (int, int) {
		defer cleanup()
		count += delta
		return count, value
	}
	evaluations := 0
	p, original := fn(func() int {
		evaluations++
		return 3
	}())
	q, _ := fn(5)
	if evaluations != 1 {
		panic("argument was evaluated more than once")
	}
	return p, q, original
}

//go:noinline
func selectedClosure(value, selection int) int {
	fn := func(delta int) int {
		defer cleanup()
		return value + delta
	}
	switch selection {
	case 1:
		fn = func(delta int) int { return value - delta }
	case 2:
		fn = nil
	}
	return fn(3)
}

//go:noinline
func genericClosure[T ~int](value T) T {
	fn := func(delta T) T {
		defer cleanup()
		return value + delta
	}
	return fn(3)
}

func main() {
	p, q, original := twoContexts(4)
	if p != 7 || q != 12 || original != 4 {
		panic("call read the wrong closure context")
	}
	p, q, original = twoContexts(9)
	if p != 12 || q != 17 || original != 9 {
		panic("call reused an old closure context")
	}
	if selectedClosure(4, 0) != 7 || selectedClosure(4, 1) != 1 {
		panic("call ignored a reassigned function value")
	}
	func() {
		defer func() {
			if recover() == nil {
				panic("nil function call returned")
			}
		}()
		selectedClosure(4, 2)
	}()
	if genericClosure(8) != 11 {
		panic("call read the wrong generic closure context")
	}
}
