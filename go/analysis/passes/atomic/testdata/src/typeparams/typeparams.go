// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the atomic checker.

package a

import (
	"sync/atomic"
)

type Subtractable interface {
	~int64
}

func Sub[T Subtractable](addr *T, delta T) T {
	// the followings result in type errors, but doesn't stop this vet check
	*addr = atomic.AddInt64(addr, -delta)  // want "direct assignment to atomic value"
	*addr = atomic.AddUintptr(addr, delta) // want "direct assignment to atomic value"
	atomic.AddInt64()  // vet ignores it
	return *addr
}

type _S[T Subtractable] struct {
	x *T
}

func (v _S) AddInt64(_ *int64, delta int64) int64 {
	*v.x = atomic.AddInt64(v.x, delta)  // want "direct assignment to atomic value"
	return *v.x
}

func NonAtomicInt64() {
	var atomic _S[int64]
	*atomic.x = atomic.AddInt64(atomic.x, 123)  // ok; AddInt64 is not sync/atomic.AddInt64.
}