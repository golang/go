// errorcheck -0 -d=ssa/intrinsics/debug
// +build amd64 arm64 mips mipsle mips64 mips64le ppc64 ppc64le riscv64 s390x

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "sync/atomic"

var x uint32

func atomics() {
	_ = atomic.LoadUint32(&x)             // ERROR "intrinsic substitution for LoadUint32"
	atomic.StoreUint32(&x, 1)             // ERROR "intrinsic substitution for StoreUint32"
	atomic.AddUint32(&x, 1)               // ERROR "intrinsic substitution for AddUint32"
	atomic.SwapUint32(&x, 1)              // ERROR "intrinsic substitution for SwapUint32"
	atomic.CompareAndSwapUint32(&x, 1, 2) // ERROR "intrinsic substitution for CompareAndSwapUint32"
}
