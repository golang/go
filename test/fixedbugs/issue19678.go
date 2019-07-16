// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Used to crash when compiling functions containing
// forward refs in dead code.

package p

var f func(int)

func g() {
l1:
	i := 0
	goto l1
l2:
	f(i)
	goto l2
}
