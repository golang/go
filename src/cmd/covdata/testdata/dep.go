// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dep

func Dep1() int {
	return 42
}

func PDep(x int) {
	if x != 1010101 {
		println(x)
	} else {
		panic("bad")
	}
}
