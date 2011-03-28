// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package atomic provides low-level atomic memory primitives
// useful for implementing synchronization algorithms.
//
// These functions require great care to be used correctly.
// Except for special, low-level applications, synchronization is better
// done with channels or the facilities of the sync package.
// Share memory by communicating;
// don't communicate by sharing memory.
//
// The compare-and-swap operation, implemented by the CompareAndSwapT
// functions, is the atomic equivalent of:
//
//	if *val == old {
//		*val = new
//		return true
//	}
//	return false
//
package atomic

// BUG(rsc): On ARM, the 64-bit functions use instructions unavailable before ARM 11.
//
// On x86-32, the 64-bit functions use instructions unavailable before the Pentium.

// CompareAndSwapInt32 executes the compare-and-swap operation for an int32 value.
func CompareAndSwapInt32(val *int32, old, new int32) (swapped bool)

// CompareAndSwapInt64 executes the compare-and-swap operation for an int64 value.
func CompareAndSwapInt64(val *int64, old, new int64) (swapped bool)

// CompareAndSwapUint32 executes the compare-and-swap operation for a uint32 value.
func CompareAndSwapUint32(val *uint32, old, new uint32) (swapped bool)

// CompareAndSwapUint64 executes the compare-and-swap operation for a uint64 value.
func CompareAndSwapUint64(val *uint64, old, new uint64) (swapped bool)

// CompareAndSwapUintptr executes the compare-and-swap operation for a uintptr value.
func CompareAndSwapUintptr(val *uintptr, old, new uintptr) (swapped bool)

// AddInt32 atomically adds delta to *val and returns the new value.
func AddInt32(val *int32, delta int32) (new int32)

// AddUint32 atomically adds delta to *val and returns the new value.
func AddUint32(val *uint32, delta uint32) (new uint32)

// AddInt64 atomically adds delta to *val and returns the new value.
func AddInt64(val *int64, delta int64) (new int64)

// AddUint64 atomically adds delta to *val and returns the new value.
func AddUint64(val *uint64, delta uint64) (new uint64)

// AddUintptr atomically adds delta to *val and returns the new value.
func AddUintptr(val *uintptr, delta uintptr) (new uintptr)

// Helper for ARM.  Linker will discard on other systems
func panic64() {
	panic("sync/atomic: broken 64-bit atomic operations (buggy QEMU)")
}
