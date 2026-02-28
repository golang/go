// compile

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package foo

type T interface {
	foo()
}

func f() (T, int)

func g(v interface{}) (interface{}, int) {
	var x int
	v, x = f()
	return v, x
}
