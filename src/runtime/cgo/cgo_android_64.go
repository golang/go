// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && android && (amd64 || arm64)

package cgo

// cSigaction matches Bionic's struct sigaction on 64-bit Android (LP64).
type cSigaction struct {
	flags    int32
	handler  uintptr
	mask     [1]uintptr
	restorer uintptr
}
