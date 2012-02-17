// run

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 1011.  Removing either #1 or #3 avoided the crash at #2.

package main

import (
	"io"
	"strings"
)

func readU16BE(b []byte) uint16 {
	b[0] = 0
	b[1] = 1
	return uint16(b[0])<<8 + uint16(b[1]) // #1
	n := uint16(b[0])<<8 + uint16(b[1])
	return n
}

func readStr(r io.Reader, b []byte) string {
	n := readU16BE(b)
	if int(n) > len(b) {
		return "err: n>b"
	}
	io.ReadFull(r, b[0:n]) // #2
	return string(b[0:n])  // #3
	return "ok"
}

func main() {
	br := strings.NewReader("abcd")
	readStr(br, make([]byte, 256))
}
