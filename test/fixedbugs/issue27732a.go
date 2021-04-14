// errorcheck -0 -m -l -smallframes

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This checks that the -smallframes flag forces a large variable to heap.

package main

const (
	bufferLen = 200000
)

type kbyte []byte
type circularBuffer [bufferLen]kbyte

var sink byte

func main() {
	var c circularBuffer // ERROR "moved to heap: c$"
	sink = c[0][0]
}
