// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func F() {
	var x struct {
		x *int
		w [1e9][1e9][1e9][0]*int
		y *int
	}
	println(&x)
}
