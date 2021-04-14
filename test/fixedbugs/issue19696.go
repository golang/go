// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Used to crash when compiling assignments involving [0]T,
// where T is not SSA-able.

package p

type s struct {
	a, b, c, d, e int
}

func f() {
	var i int
	arr := [0]s{}
	arr[i].a++
}
