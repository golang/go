// errorcheck

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _(a, b, c int) {
	_ = a
	_ = a, b    // ERROR "assignment mismatch: 1 variable but 2 values"
	_ = a, b, c // ERROR "assignment mismatch: 1 variable but 3 values"

	_, _ = a // ERROR "assignment mismatch: 2 variables but 1 value"
	_, _ = a, b
	_, _ = a, b, c // ERROR "assignment mismatch: 2 variables but 3 values"

	_, _, _ = a    // ERROR "assignment mismatch: 3 variables but 1 value"
	_, _, _ = a, b // ERROR "assignment mismatch: 3 variables but 2 values"
	_, _, _ = a, b, c
}

func f1() int
func f2() (int, int)
func f3() (int, int, int)

func _() {
	_ = f1()
	_ = f2() // ERROR "assignment mismatch: 1 variable but f2 returns 2 values"
	_ = f3() // ERROR "assignment mismatch: 1 variable but f3 returns 3 values"

	_, _ = f1() // ERROR "assignment mismatch: 2 variables but f1 returns 1 value"
	_, _ = f2()
	_, _ = f3() // ERROR "assignment mismatch: 2 variables but f3 returns 3 values"

	_, _, _ = f1() // ERROR "assignment mismatch: 3 variables but f1 returns 1 value"
	_, _, _ = f2() // ERROR "assignment mismatch: 3 variables but f2 returns 2 values"
	_, _, _ = f3()

	// test just a few := cases as they use the same code as the = case
	a1 := f3()         // ERROR "assignment mismatch: 1 variable but f3 returns 3 values"
	a2, b2 := f1()     // ERROR "assignment mismatch: 2 variables but f1 returns 1 value"
	a3, b3, c3 := f2() // ERROR "assignment mismatch: 3 variables but f2 returns 2 values"
}

type T struct{}

func (T) f1() int
func (T) f2() (int, int)
func (T) f3() (int, int, int)

func _(x T) {
	_ = x.f1()
	_ = x.f2() // ERROR "assignment mismatch: 1 variable but .\.f2 returns 2 values"
	_ = x.f3() // ERROR "assignment mismatch: 1 variable but .\.f3 returns 3 values"

	_, _ = x.f1() // ERROR "assignment mismatch: 2 variables but .\.f1 returns 1 value"
	_, _ = x.f2()
	_, _ = x.f3() // ERROR "assignment mismatch: 2 variables but .\.f3 returns 3 values"

	_, _, _ = x.f1() // ERROR "assignment mismatch: 3 variables but .\.f1 returns 1 value"
	_, _, _ = x.f2() // ERROR "assignment mismatch: 3 variables but .\.f2 returns 2 values"
	_, _, _ = x.f3()

	// test just a few := cases as they use the same code as the = case
	a1 := x.f3()         // ERROR "assignment mismatch: 1 variable but .\.f3 returns 3 values"
	a2, b2 := x.f1()     // ERROR "assignment mismatch: 2 variables but .\.f1 returns 1 value"
	a3, b3, c3 := x.f2() // ERROR "assignment mismatch: 3 variables but .\.f2 returns 2 values"
}

// some one-off cases
func _() {
	_ = (f2)
	_ = f1(), 2         // ERROR "assignment mismatch: 1 variable but 2 values"
	_, _ = (f1()), f2() // ERROR "multiple-value f2\(\) .*in single-value context"
	_, _, _ = f3(), 3   // ERROR "assignment mismatch: 3 variables but 2 values|multiple-value f3\(\) .*in single-value context"
}
