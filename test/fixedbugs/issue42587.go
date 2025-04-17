// compile

// Copyright 2020 The Go Authors. All rights reserved.  Use of this
// source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package p

func f() {
	var i, j int
	_ = func() {
		i = 32
		j = j>>i | len([]int{})
	}
}
