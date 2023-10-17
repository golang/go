// run

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test making channels of a zero-sized type.

package main

func main() {
	_ = make(chan [0]byte)
	_ = make(chan [0]byte, 1)
	_ = make(chan struct{})
	_ = make(chan struct{}, 1)
}
