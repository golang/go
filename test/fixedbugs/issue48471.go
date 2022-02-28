// errorcheck -G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type I interface{ M(int) }

type T struct{}

type T2 struct{}

func (*T2) m(int)

type T3 struct{}

func (*T3) M(string) {}

type T4 struct{}

func (*T4) M(int)

type T5 struct{}

func (T5) m(int) {}

type T6 struct{}

func (T6) m(int) string { return "" }

func f(I)

func g() {
	f(new(T)) // ERROR "cannot use new\(T\) \(.*type \*T\) as type I in argument to f:\n\t\*T does not implement I \(missing M method\)"

	var i I
	i = new(T)    // ERROR "cannot use new\(T\) \(.*type \*T\) as type I in assignment:\n\t\*T does not implement I \(missing M method\)"
	i = I(new(T)) // ERROR "cannot convert new\(T\) \(.*type \*T\) to type I:\n\t\*T does not implement I \(missing M method\)"
	i = new(T2)   // ERROR "cannot use new\(T2\) \(.*type \*T2\) as type I in assignment:\n\t\*T2 does not implement I \(missing M method\)\n\t\thave m\(int\)\n\t\twant M\(int\)"

	i = new(T3) // ERROR "cannot use new\(T3\) \(.*type \*T3\) as type I in assignment:\n\t\*T3 does not implement I \(wrong type for M method\)\n\t\thave M\(string\)\n\t\twant M\(int\)"

	i = T4{}   // ERROR "cannot use T4\{\} \(.*type T4\) as type I in assignment:\n\tT4 does not implement I \(M method has pointer receiver\)"
	i = new(I) // ERROR "cannot use new\(I\) \(.*type \*I\) as type I in assignment:\n\t\*I does not implement I \(type \*I is pointer to interface, not interface\)"

	_ = i.(*T2) // ERROR "impossible type assertion: i.\(\*T2\)\n\t\*T2 does not implement I \(missing M method\)\n\t\thave m\(int\)\n\t\twant M\(int\)"
	_ = i.(*T3) // ERROR "impossible type assertion: i.\(\*T3\)\n\t\*T3 does not implement I \(wrong type for M method\)\n\t\thave M\(string\)\n\t\twant M\(int\)"
	_ = i.(T5)  // ERROR ""impossible type assertion: i.\(T5\)\n\tT5 does not implement I \(missing M method\)\n\t\thave m\(int\)\n\t\twant M\(int\)"
	_ = i.(T6)  // ERROR "impossible type assertion: i.\(T6\)\n\tT6 does not implement I \(missing M method\)\n\t\thave m\(int\) string\n\t\twant M\(int\)"

	var t *T4
	t = i // ERROR "cannot use i \(variable of type I\) as type \*T4 in assignment:\n\tneed type assertion"
	_ = i
}
