// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !386,!amd64,!amd64p32,!s390x,!arm,!arm64,!ppc64,!ppc64le,!mips,!mipsle,!mips64,!mips64le,!wasm

package bytealg

import _ "unsafe" // for go:linkname

func IndexByte(b []byte, c byte) int {
	for i, x := range b {
		if x == c {
			return i
		}
	}
	return -1
}

func IndexByteString(s string, c byte) int {
	for i := 0; i < len(s); i++ {
		if s[i] == c {
			return i
		}
	}
	return -1
}

//go:linkname bytes_IndexByte bytes.IndexByte
func bytes_IndexByte(b []byte, c byte) int {
	for i, x := range b {
		if x == c {
			return i
		}
	}
	return -1
}

//go:linkname strings_IndexByte strings.IndexByte
func strings_IndexByte(s string, c byte) int {
	for i := 0; i < len(s); i++ {
		if s[i] == c {
			return i
		}
	}
	return -1
}
