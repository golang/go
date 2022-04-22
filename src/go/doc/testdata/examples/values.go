// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package foo_test

// Variable declaration with fewer values than names.

func f() (int, int) {
	return 1, 2
}

var a, b = f()

// Need two examples to hit playExample.

func ExampleA() {
	_ = a
}

func ExampleB() {
}
