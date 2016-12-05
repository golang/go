// errorcheck

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func foo() (int, int) {
	return 2.3 // ERROR "not enough arguments to return\n\thave \(number\)\n\twant \(int, int\)"
}

func foo2() {
	return int(2), 2 // ERROR "too many arguments to return\n\thave \(int, number\)\n\twant \(\)"
}

func foo3(v int) (a, b, c, d int) {
	if v >= 0 {
		return 1 // ERROR "not enough arguments to return\n\thave \(number\)\n\twant \(int, int, int, int\)"
	}
	return 2, 3 // ERROR "not enough arguments to return\n\thave \(number, number\)\n\twant \(int, int, int, int\)"
}

func foo4(name string) (string, int) {
	switch name {
	case "cow":
		return "moo" // ERROR "not enough arguments to return\n\thave \(string\)\n\twant \(string, int\)"
	case "dog":
		return "dog", 10, true // ERROR "too many arguments to return\n\thave \(string, number, bool\)\n\twant \(string, int\)"
	case "fish":
		return "" // ERROR "not enough arguments to return\n\thave \(string\)\n\twant \(string, int\)"
	default:
		return "lizard", 10
	}
}

type S int
type T string
type U float64

func foo5() (S, T, U) {
	if false {
		return "" // ERROR "not enough arguments to return\n\thave \(string\)\n\twant \(S, T, U\)"
	} else {
		ptr := new(T)
		return ptr // ERROR "not enough arguments to return\n\thave \(\*T\)\n\twant \(S, T, U\)"
	}
	return new(S), 12.34, 1 + 0i, 'r', true // ERROR "too many arguments to return\n\thave \(\*S, number, number, number, bool\)\n\twant \(S, T, U\)"
}

func foo6() (T, string) {
	return "T", true, true // ERROR "too many arguments to return\n\thave \(string, bool, bool\)\n\twant \(T, string\)"
}
