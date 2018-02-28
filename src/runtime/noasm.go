// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Routines that are implemented in assembly in asm_{amd64,386,arm,arm64,ppc64x,s390x}.s
// These routines have corresponding stubs in stubs_asm.go.

// +build mips64 mips64le

package runtime

import _ "unsafe" // for go:linkname

func cmpstring(s1, s2 string) int {
	l := len(s1)
	if len(s2) < l {
		l = len(s2)
	}
	for i := 0; i < l; i++ {
		c1, c2 := s1[i], s2[i]
		if c1 < c2 {
			return -1
		}
		if c1 > c2 {
			return +1
		}
	}
	if len(s1) < len(s2) {
		return -1
	}
	if len(s1) > len(s2) {
		return +1
	}
	return 0
}

//go:linkname bytes_Compare bytes.Compare
func bytes_Compare(s1, s2 []byte) int {
	l := len(s1)
	if len(s2) < l {
		l = len(s2)
	}
	if l == 0 || &s1[0] == &s2[0] {
		goto samebytes
	}
	for i := 0; i < l; i++ {
		c1, c2 := s1[i], s2[i]
		if c1 < c2 {
			return -1
		}
		if c1 > c2 {
			return +1
		}
	}
samebytes:
	if len(s1) < len(s2) {
		return -1
	}
	if len(s1) > len(s2) {
		return +1
	}
	return 0
}
