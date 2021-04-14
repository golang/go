// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !amd64,!arm,!arm64,!ppc64le,!ppc64,!riscv64,!s390x

package bytealg

func Count(b []byte, c byte) int {
	n := 0
	for _, x := range b {
		if x == c {
			n++
		}
	}
	return n
}

func CountString(s string, c byte) int {
	n := 0
	for i := 0; i < len(s); i++ {
		if s[i] == c {
			n++
		}
	}
	return n
}
