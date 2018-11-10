// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p2

import "./p1"

type T struct {
	p1.T
}

func F() {
	var t T
	p1.F(&t.T)
}
