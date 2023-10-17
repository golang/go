// run -gcflags=-d=ssa/check/on

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// As of 2019-06, bug affects/ed amd64 and s390x.

package main

var foo = []byte{105, 57, 172, 152}

func main() {
	for i := 0; i < len(foo); i += 4 {
		// Requires inlining and non-constant i
		// Note the bug/fix also apply to different widths, but was unable to reproduce for those.
		println(readLittleEndian32_2(foo[i], foo[i+1], foo[i+2], foo[i+3]))
	}
}

func readLittleEndian32_2(a, b, c, d byte) uint32 {
	return uint32(a) | (uint32(b) << 8) | (uint32(c) << 16) | (uint32(d) << 24)
}
