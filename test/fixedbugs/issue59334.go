// run -tags=purego -gcflags=all=-d=checkptr

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "crypto/subtle"

func main() {
	dst := make([]byte, 5)
	src := make([]byte, 5)
	for _, n := range []int{1024, 2048} { // just to make the size non-constant
		b := make([]byte, n)
		subtle.XORBytes(dst, src, b[n-5:])
	}
}
