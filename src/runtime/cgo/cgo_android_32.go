// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && android && 386

package cgo

// cSigaction matches Bionic's struct sigaction on 32-bit Android (ILP32).
type cSigaction struct {
	handler  uintptr
	mask     [1]uintptr
	flags    int32
	restorer uintptr
}
