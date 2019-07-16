// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Used to crash when compiling assignments involving [0]T,
// where T is not SSA-able.

package a

func f() {
	var i int
	arr := [0][2]int{}
	arr[i][0] = 0
}
