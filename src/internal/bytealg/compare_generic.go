// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !386 && !amd64 && !s390x && !arm && !arm64 && !loong64 && !ppc64 && !ppc64le && !mips && !mipsle && !wasm && !mips64 && !mips64le && !riscv64

package bytealg

import _ "unsafe" // for go:linkname

func Compare(a, b []byte) int {
	l := len(a)
	if len(b) < l {
		l = len(b)
	}
	if l == 0 || &a[0] == &b[0] {
		goto samebytes
	}
	for i := 0; i < l; i++ {
		c1, c2 := a[i], b[i]
		if c1 < c2 {
			return -1
		}
		if c1 > c2 {
			return +1
		}
	}
samebytes:
	if len(a) < len(b) {
		return -1
	}
	if len(a) > len(b) {
		return +1
	}
	return 0
}

func CompareString(a, b string) int {
	return runtime_cmpstring(a, b)
}

// runtime.cmpstring calls are emitted by the compiler.
//
// runtime.cmpstring should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/bytedance/gopkg
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname runtime_cmpstring runtime.cmpstring
func runtime_cmpstring(a, b string) int {
	l := len(a)
	if len(b) < l {
		l = len(b)
	}
	for i := 0; i < l; i++ {
		c1, c2 := a[i], b[i]
		if c1 < c2 {
			return -1
		}
		if c1 > c2 {
			return +1
		}
	}
	if len(a) < len(b) {
		return -1
	}
	if len(a) > len(b) {
		return +1
	}
	return 0
}
