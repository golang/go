// errorcheck

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f0()
func f1(_ int)
func f2(_, _ int)
func f2ddd(_, _ int, _ ...int)

func f() {
	var x int
	f0(1)              // ERROR "too many arguments in call to f0\n\thave \(number\)\n\twant \(\)"
	f0(x)              // ERROR "too many arguments in call to f0\n\thave \(int\)\n\twant \(\)"
	f1()               // ERROR "not enough arguments in call to f1\n\thave \(\)\n\twant \(int\)"
	f1(1, 2)           // ERROR "too many arguments in call to f1\n\thave \(number, number\)\n\twant \(int\)"
	f2(1)              // ERROR "not enough arguments in call to f2\n\thave \(number\)\n\twant \(int, int\)"
	f2(1, "foo", true) // ERROR "too many arguments in call to f2\n\thave \(number, string, bool\)\n\twant \(int, int\)"
	f2ddd(1)           // ERROR "not enough arguments in call to f2ddd\n\thave \(number\)\n\twant \(int, int, \.\.\.int\)"
	f2ddd(1, 2)
	f2ddd(1, 2, 3)
}
