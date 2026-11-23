// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && linux && (loong64 || riscv64)

package cgo

// goSigaction matches runtime.sigactiont on linux/loong64 and linux/riscv64.
type goSigaction struct {
	handler  uintptr
	flags    uintptr
	mask     uint64
	restorer uintptr
}
