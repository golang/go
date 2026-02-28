// skip

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var a [1<<31 - 1024]byte

func main() {
	if a[0] != 0 {
		panic("bad array")
	}
}
