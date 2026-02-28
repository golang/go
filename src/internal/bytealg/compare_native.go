// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build 386 || amd64 || s390x || arm || arm64 || loong64 || ppc64 || ppc64le || mips || mipsle || wasm || mips64 || mips64le || riscv64

package bytealg

import _ "unsafe" // For go:linkname

//go:noescape
func Compare(a, b []byte) int

func CompareString(a, b string) int {
	return abigen_runtime_cmpstring(a, b)
}

// The declaration below generates ABI wrappers for functions
// implemented in assembly in this package but declared in another
// package.

//go:linkname abigen_runtime_cmpstring runtime.cmpstring
func abigen_runtime_cmpstring(a, b string) int
