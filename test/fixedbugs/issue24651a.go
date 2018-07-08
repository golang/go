//errorcheck -0 -race -m -m

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:norace
func Foo(x int) int { // ERROR "cannot inline Foo: marked go:norace with -race compilation$"
	return x * (x + 1) * (x + 2)
}

func Bar(x int) int { // ERROR "can inline Bar as: func\(int\) int { return x \* \(x \+ 1\) \* \(x \+ 2\) }$"
	return x * (x + 1) * (x + 2)
}

var x = 5

//go:noinline Provide a clean, constant reason for not inlining main
func main() { // ERROR "cannot inline main: marked go:noinline$"
	println("Foo(", x, ")=", Foo(x))
	println("Bar(", x, ")=", Bar(x)) // ERROR "inlining call to Bar func\(int\) int { return x \* \(x \+ 1\) \* \(x \+ 2\) }$"
}
