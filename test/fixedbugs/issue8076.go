// compile

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 8076. nilwalkfwd walked forward forever
// on the instruction loop following the dereference.

package main

func main() {
	_ = *(*int)(nil)
L:
	_ = 0
	goto L
}
