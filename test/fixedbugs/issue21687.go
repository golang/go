// run

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 21687: cmd/compile evaluates x twice in "x op= y", which was
// detectable if evaluating y affects x.

package main

func ptrs() (int, int) {
	one := 1
	two := 2

	x := &one
	*x += func() int {
		x = &two
		return 0
	}()

	return one, two
}

func slices() (int, int) {
	one := []int{1}
	two := []int{2}

	x := one
	x[0] += func() int {
		x = two
		return 0
	}()

	return one[0], two[0]
}

func maps() (int, int) {
	one := map[int]int{0: 1}
	two := map[int]int{0: 2}

	x := one
	x[0] += func() int {
		x = two
		return 0
	}()

	return one[0], two[0]
}

var tests = [...]func() (int, int){
	ptrs,
	slices,
	maps,
}

func main() {
	bad := 0
	for i, f := range tests {
		if a, b := f(); a+b != 3 {
			println(i, a, b)
			bad++
		}
	}
	if bad != 0 {
		panic(bad)
	}
}
