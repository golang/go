// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build 386 amd64 amd64p32 s390x arm arm64 ppc64 ppc64le mips mipsle wasm

package bytealg

//go:noescape
func Compare(a, b []byte) int
