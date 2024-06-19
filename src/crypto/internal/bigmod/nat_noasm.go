// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build purego || !(386 || amd64 || arm || arm64 || loong64 || ppc64 || ppc64le || riscv64 || s390x)

package bigmod

import "unsafe"

func addMulVVW1024(z, x *uint, y uint) (c uint) {
	return addMulVVW(unsafe.Slice(z, 1024/_W), unsafe.Slice(x, 1024/_W), y)
}

func addMulVVW1536(z, x *uint, y uint) (c uint) {
	return addMulVVW(unsafe.Slice(z, 1536/_W), unsafe.Slice(x, 1536/_W), y)
}

func addMulVVW2048(z, x *uint, y uint) (c uint) {
	return addMulVVW(unsafe.Slice(z, 2048/_W), unsafe.Slice(x, 2048/_W), y)
}
