// compile

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 20923: gccgo failed to compile parenthesized select case expressions.

package p

func F(c chan bool) {
	select {
	case (<-c):
	case _ = (<-c):
	case _, _ = (<-c):
	case (c) <- true:
	default:
	}
}
