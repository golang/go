// errorcheck

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 7675: fewer errors for wrong argument count

package p

func f(string, int, float64, string)

func g(string, int, float64, ...string)

func main() {
	f(1, 0.5, "hello") // ERROR "not enough arguments"
	f("1", 2, 3.1, "4")
	f(1, 0.5, "hello", 4, 5) // ERROR "too many arguments"
	g(1, 0.5)                // ERROR "not enough arguments"
	g("1", 2, 3.1)
	g(1, 0.5, []int{3, 4}...) // ERROR "not enough arguments"
	g("1", 2, 3.1, "4", "5")
	g(1, 0.5, "hello", 4, []int{5, 6}...) // ERROR "too many arguments"
}
