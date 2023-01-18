// errorcheck -0 -m -l

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package foo

import "sync/atomic"

func AddInt32(x *int32) { // ERROR "x does not escape$"
	atomic.AddInt32(x, 42)
}
func AddUint32(x *uint32) { // ERROR "x does not escape$"
	atomic.AddUint32(x, 42)
}
func AddUintptr(x *uintptr) { // ERROR "x does not escape$"
	atomic.AddUintptr(x, 42)
}

func CompareAndSwapInt32(x *int32) { // ERROR "x does not escape$"
	atomic.CompareAndSwapInt32(x, 42, 42)
}
func CompareAndSwapUint32(x *uint32) { // ERROR "x does not escape$"
	atomic.CompareAndSwapUint32(x, 42, 42)
}
func CompareAndSwapUintptr(x *uintptr) { // ERROR "x does not escape$"
	atomic.CompareAndSwapUintptr(x, 42, 42)
}

func LoadInt32(x *int32) { // ERROR "x does not escape$"
	atomic.LoadInt32(x)
}
func LoadUint32(x *uint32) { // ERROR "x does not escape$"
	atomic.LoadUint32(x)
}
func LoadUintptr(x *uintptr) { // ERROR "x does not escape$"
	atomic.LoadUintptr(x)
}

func StoreInt32(x *int32) { // ERROR "x does not escape$"
	atomic.StoreInt32(x, 42)
}
func StoreUint32(x *uint32) { // ERROR "x does not escape$"
	atomic.StoreUint32(x, 42)
}
func StoreUintptr(x *uintptr) { // ERROR "x does not escape$"
	atomic.StoreUintptr(x, 42)
}

func SwapInt32(x *int32) { // ERROR "x does not escape$"
	atomic.SwapInt32(x, 42)
}
func SwapUint32(x *uint32) { // ERROR "x does not escape$"
	atomic.SwapUint32(x, 42)
}
func SwapUintptr(x *uintptr) { // ERROR "x does not escape$"
	atomic.SwapUintptr(x, 42)
}
