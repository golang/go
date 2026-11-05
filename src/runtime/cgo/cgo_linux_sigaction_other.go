// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && linux && (386 || amd64 || arm64 || ppc64 || ppc64le)

package cgo

// goSigaction matches runtime.sigactiont on linux/{386,amd64,arm64,ppc64,ppc64le}.
type goSigaction struct {
	handler  uintptr
	flags    uintptr
	restorer uintptr
	mask     uint64
}
