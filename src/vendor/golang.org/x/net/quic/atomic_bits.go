// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import "sync/atomic"

// atomicBits is an atomic uint32 that supports setting individual bits.
type atomicBits[T ~uint32] struct {
	bits atomic.Uint32
}

// set sets the bits in mask to the corresponding bits in v.
// It returns the new value.
func (a *atomicBits[T]) set(v, mask T) T {
	if v&^mask != 0 {
		panic("BUG: bits in v are not in mask")
	}
	for {
		o := a.bits.Load()
		n := (o &^ uint32(mask)) | uint32(v)
		if a.bits.CompareAndSwap(o, n) {
			return T(n)
		}
	}
}

func (a *atomicBits[T]) load() T {
	return T(a.bits.Load())
}
