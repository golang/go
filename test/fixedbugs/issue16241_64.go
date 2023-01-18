//go:build !(386 || arm || mips || mipsle)

// errorcheck -0 -m -l

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package foo

import "sync/atomic"

func AddInt64(x *int64) { // ERROR "x does not escape$"
	atomic.AddInt64(x, 42)
}
func AddUint64(x *uint64) { // ERROR "x does not escape$"
	atomic.AddUint64(x, 42)
}

func CompareAndSwapInt64(x *int64) { // ERROR "x does not escape$"
	atomic.CompareAndSwapInt64(x, 42, 42)
}
func CompareAndSwapUint64(x *uint64) { // ERROR "x does not escape$"
	atomic.CompareAndSwapUint64(x, 42, 42)
}

func LoadInt64(x *int64) { // ERROR "x does not escape$"
	atomic.LoadInt64(x)
}
func LoadUint64(x *uint64) { // ERROR "x does not escape$"
	atomic.LoadUint64(x)
}

func StoreInt64(x *int64) { // ERROR "x does not escape$"
	atomic.StoreInt64(x, 42)
}
func StoreUint64(x *uint64) { // ERROR "x does not escape$"
	atomic.StoreUint64(x, 42)
}

func SwapInt64(x *int64) { // ERROR "x does not escape$"
	atomic.SwapInt64(x, 42)
}
func SwapUint64(x *uint64) { // ERROR "x does not escape$"
	atomic.SwapUint64(x, 42)
}
