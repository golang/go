// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !386 && !amd64 && !arm64 && !arm && !mips && !mipsle && !ppc64 && !ppc64le && !s390x && !riscv64 && !wasm

package math

const haveArchSqrt = false

func archSqrt(x float64) float64 {
	panic("not implemented")
}
