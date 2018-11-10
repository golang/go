// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Use the functions in one.go so that the inlined
// forms get type-checked.

package pkg2

import "./one"

func use() {
	one.F1(nil)
	one.F2(nil)
	one.F3()
	one.F4(1)

	var t *one.T
	t.M()
	t.MM()
}

var V = []one.PB{{}, {}}

func F() *one.PB
