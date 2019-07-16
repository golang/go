// errorcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 23732: Give better details about which struct
// initializer has the wrong number of values.

package main

type Foo struct {
	A int
	B int
	C interface{}
	Bar
}

type Bar struct {
	A string
}

func main() {
	_ = Foo{
		1,
		2,
		3, // ERROR "too few values in Foo literal"
	}

	_ = Foo{
		1,
		2,
		3,
		Bar{"A", "B"}, // ERROR "too many values in Bar literal"
	}

	_ = Foo{
		1,
		2,
		Bar{"A", "B"}, // ERROR "too many values in Bar literal" "too few values in Foo literal"
	}
}
