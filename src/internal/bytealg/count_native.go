// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 || arm || arm64 || ppc64le || ppc64 || riscv64 || s390x

package bytealg

//go:noescape
func Count(b []byte, c byte) int

//go:noescape
func CountString(s string, c byte) int

// A backup implementation to use by assembly.
func countGeneric(b []byte, c byte) int {
	n := 0
	for _, x := range b {
		if x == c {
			n++
		}
	}
	return n
}
func countGenericString(s string, c byte) int {
	n := 0
	for i := 0; i < len(s); i++ {
		if s[i] == c {
			n++
		}
	}
	return n
}
