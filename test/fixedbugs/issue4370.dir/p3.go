// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p3

import "./p2"

func F() {
	p2.F()
	var t p2.T
	println(t.T.M())
}
