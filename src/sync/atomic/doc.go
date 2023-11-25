// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package atomic provides low-level atomic memory primitives
// useful for implementing synchronization algorithms.
//
// These functions require great care to be used correctly.
// Except for special, low-level applications, synchronization is better
// done with channels or the facilities of the [sync] package.
// Share memory by communicating;
// don't communicate by sharing memory.
//
// The swap operation, implemented by the SwapT functions, is the atomic
// equivalent of:
//
//	old = *addr
//	*addr = new
//	return old
//
// The compare-and-swap operation, implemented by the CompareAndSwapT
// functions, is the atomic equivalent of:
//
//	if *addr == old {
//		*addr = new
//		return true
//	}
//	return false
//
// The add operation, implemented by the AddT functions, is the atomic
// equivalent of:
//
//	*addr += delta
//	return *addr
//
// The load and store operations, implemented by the LoadT and StoreT
// functions, are the atomic equivalents of "return *addr" and
// "*addr = val".
//
// In the terminology of the Go memory model, if the effect of
// an atomic operation A is observed by atomic operation B,
// then A “synchronizes before” B.
// Additionally, all the atomic operations executed in a program
// behave as though executed in some sequentially consistent order.
// This definition provides the same semantics as
// C++'s sequentially consistent atomics and Java's volatile variables.
package atomic

import (
	"unsafe"
)

// BUG(rsc): On 386, the 64-bit functions use instructions unavailable before the Pentium MMX.
//
// On non-Linux ARM, the 64-bit functions use instructions unavailable before the ARMv6k core.
//
// On ARM, 386, and 32-bit MIPS, it is the caller's responsibility to arrange
// for 64-bit alignment of 64-bit words accessed atomically via the primitive
// atomic functions (types [Int64] and [Uint64] are automatically aligned).
// The first word in an allocated struct, array, or slice; in a global
// variable; or in a local variable (because the subject of all atomic operations
// will escape to the heap) can be relied upon to be 64-bit aligned.

// SwapInt32 atomically stores new into *addr and returns the previous *addr value.
// Consider using the more ergonomic and less error-prone [Int32.Swap] instead.
func SwapInt32(addr *int32, new int32) (old int32)

// SwapInt64 atomically stores new into *addr and returns the previous *addr value.
// Consider using the more ergonomic and less error-prone [Int64.Swap] instead
// (particularly if you target 32-bit platforms; see the bugs section).
func SwapInt64(addr *int64, new int64) (old int64)

// SwapUint32 atomically stores new into *addr and returns the previous *addr value.
// Consider using the more ergonomic and less error-prone [Uint32.Swap] instead.
func SwapUint32(addr *uint32, new uint32) (old uint32)

// SwapUint64 atomically stores new into *addr and returns the previous *addr value.
// Consider using the more ergonomic and less error-prone [Uint64.Swap] instead
// (particularly if you target 32-bit platforms; see the bugs section).
func SwapUint64(addr *uint64, new uint64) (old uint64)

// SwapUintptr atomically stores new into *addr and returns the previous *addr value.
// Consider using the more ergonomic and less error-prone [Uintptr.Swap] instead.
func SwapUintptr(addr *uintptr, new uintptr) (old uintptr)

// SwapPointer atomically stores new into *addr and returns the previous *addr value.
// Consider using the more ergonomic and less error-prone [Pointer.Swap] instead.
func SwapPointer(addr *unsafe.Pointer, new unsafe.Pointer) (old unsafe.Pointer)

// CompareAndSwapInt32 executes the compare-and-swap operation for an int32 value.
// Consider using the more ergonomic and less error-prone [Int32.CompareAndSwap] instead.
func CompareAndSwapInt32(addr *int32, old, new int32) (swapped bool)

// CompareAndSwapInt64 executes the compare-and-swap operation for an int64 value.
// Consider using the more ergonomic and less error-prone [Int64.CompareAndSwap] instead
// (particularly if you target 32-bit platforms; see the bugs section).
func CompareAndSwapInt64(addr *int64, old, new int64) (swapped bool)

// CompareAndSwapUint32 executes the compare-and-swap operation for a uint32 value.
// Consider using the more ergonomic and less error-prone [Uint32.CompareAndSwap] instead.
func CompareAndSwapUint32(addr *uint32, old, new uint32) (swapped bool)

// CompareAndSwapUint64 executes the compare-and-swap operation for a uint64 value.
// Consider using the more ergonomic and less error-prone [Uint64.CompareAndSwap] instead
// (particularly if you target 32-bit platforms; see the bugs section).
func CompareAndSwapUint64(addr *uint64, old, new uint64) (swapped bool)

// CompareAndSwapUintptr executes the compare-and-swap operation for a uintptr value.
// Consider using the more ergonomic and less error-prone [Uintptr.CompareAndSwap] instead.
func CompareAndSwapUintptr(addr *uintptr, old, new uintptr) (swapped bool)

// CompareAndSwapPointer executes the compare-and-swap operation for a unsafe.Pointer value.
// Consider using the more ergonomic and less error-prone [Pointer.CompareAndSwap] instead.
func CompareAndSwapPointer(addr *unsafe.Pointer, old, new unsafe.Pointer) (swapped bool)

// AddInt32 atomically adds delta to *addr and returns the new value.
// Consider using the more ergonomic and less error-prone [Int32.Add] instead.
func AddInt32(addr *int32, delta int32) (new int32)

// AddUint32 atomically adds delta to *addr and returns the new value.
// To subtract a signed positive constant value c from x, do AddUint32(&x, ^uint32(c-1)).
// In particular, to decrement x, do AddUint32(&x, ^uint32(0)).
// Consider using the more ergonomic and less error-prone [Uint32.Add] instead.
func AddUint32(addr *uint32, delta uint32) (new uint32)

// AddInt64 atomically adds delta to *addr and returns the new value.
// Consider using the more ergonomic and less error-prone [Int64.Add] instead
// (particularly if you target 32-bit platforms; see the bugs section).
func AddInt64(addr *int64, delta int64) (new int64)

// AddUint64 atomically adds delta to *addr and returns the new value.
// To subtract a signed positive constant value c from x, do AddUint64(&x, ^uint64(c-1)).
// In particular, to decrement x, do AddUint64(&x, ^uint64(0)).
// Consider using the more ergonomic and less error-prone [Uint64.Add] instead
// (particularly if you target 32-bit platforms; see the bugs section).
func AddUint64(addr *uint64, delta uint64) (new uint64)

// AddUintptr atomically adds delta to *addr and returns the new value.
// Consider using the more ergonomic and less error-prone [Uintptr.Add] instead.
func AddUintptr(addr *uintptr, delta uintptr) (new uintptr)

// AndInt32 atomically performs a bitwise AND operation on *addr using the bitmask provided as mask
// and returns the old value.
// Consider using the more ergonomic and less error-prone [Int32.And] instead.
func AndInt32(addr *int32, mask int32) (old int32)

// AndUint32 atomically performs a bitwise AND operation on *addr using the bitmask provided as mask
// and returns the old value.
// Consider using the more ergonomic and less error-prone [Uint32.And] instead.
func AndUint32(addr *uint32, mask uint32) (old uint32)

// AndInt64 atomically performs a bitwise AND operation on *addr using the bitmask provided as mask
// and returns the old value.
// Consider using the more ergonomic and less error-prone [Int64.And] instead.
func AndInt64(addr *int64, mask int64) (old int64)

// AndUint64 atomically performs a bitwise AND operation on *addr using the bitmask provided as mask
// and returns the old.
// Consider using the more ergonomic and less error-prone [Uint64.And] instead.
func AndUint64(addr *uint64, mask uint64) (old uint64)

// AndUintptr atomically performs a bitwise AND operation on *addr using the bitmask provided as mask
// and returns the old value.
// Consider using the more ergonomic and less error-prone [Uintptr.And] instead.
func AndUintptr(addr *uintptr, mask uintptr) (new uintptr)

// OrInt32 atomically performs a bitwise OR operation on *addr using the bitmask provided as mask
// and returns the old value.
// Consider using the more ergonomic and less error-prone [Int32.Or] instead.
func OrInt32(addr *int32, mask int32) (old int32)

// OrUint32 atomically performs a bitwise OR operation on *addr using the bitmask provided as mask
// and returns the old value.
// Consider using the more ergonomic and less error-prone [Uint32.Or] instead.
func OrUint32(addr *uint32, mask uint32) (old uint32)

// OrInt64 atomically performs a bitwise OR operation on *addr using the bitmask provided as mask
// and returns the old value.
// Consider using the more ergonomic and less error-prone [Int64.Or] instead.
func OrInt64(addr *int64, mask int64) (old int64)

// OrUint64 atomically performs a bitwise OR operation on *addr using the bitmask provided as mask
// and returns the old value.
// Consider using the more ergonomic and less error-prone [Uint64.Or] instead.
func OrUint64(addr *uint64, mask uint64) (old uint64)

// OrUintptr atomically performs a bitwise OR operation on *addr using the bitmask provided as mask
// and returns the old value.
// Consider using the more ergonomic and less error-prone [Uintptr.Or] instead.
func OrUintptr(addr *uintptr, mask uintptr) (old uintptr)

// LoadInt32 atomically loads *addr.
// Consider using the more ergonomic and less error-prone [Int32.Load] instead.
func LoadInt32(addr *int32) (val int32)

// LoadInt64 atomically loads *addr.
// Consider using the more ergonomic and less error-prone [Int64.Load] instead
// (particularly if you target 32-bit platforms; see the bugs section).
func LoadInt64(addr *int64) (val int64)

// LoadUint32 atomically loads *addr.
// Consider using the more ergonomic and less error-prone [Uint32.Load] instead.
func LoadUint32(addr *uint32) (val uint32)

// LoadUint64 atomically loads *addr.
// Consider using the more ergonomic and less error-prone [Uint64.Load] instead
// (particularly if you target 32-bit platforms; see the bugs section).
func LoadUint64(addr *uint64) (val uint64)

// LoadUintptr atomically loads *addr.
// Consider using the more ergonomic and less error-prone [Uintptr.Load] instead.
func LoadUintptr(addr *uintptr) (val uintptr)

// LoadPointer atomically loads *addr.
// Consider using the more ergonomic and less error-prone [Pointer.Load] instead.
func LoadPointer(addr *unsafe.Pointer) (val unsafe.Pointer)

// StoreInt32 atomically stores val into *addr.
// Consider using the more ergonomic and less error-prone [Int32.Store] instead.
func StoreInt32(addr *int32, val int32)

// StoreInt64 atomically stores val into *addr.
// Consider using the more ergonomic and less error-prone [Int64.Store] instead
// (particularly if you target 32-bit platforms; see the bugs section).
func StoreInt64(addr *int64, val int64)

// StoreUint32 atomically stores val into *addr.
// Consider using the more ergonomic and less error-prone [Uint32.Store] instead.
func StoreUint32(addr *uint32, val uint32)

// StoreUint64 atomically stores val into *addr.
// Consider using the more ergonomic and less error-prone [Uint64.Store] instead
// (particularly if you target 32-bit platforms; see the bugs section).
func StoreUint64(addr *uint64, val uint64)

// StoreUintptr atomically stores val into *addr.
// Consider using the more ergonomic and less error-prone [Uintptr.Store] instead.
func StoreUintptr(addr *uintptr, val uintptr)

// StorePointer atomically stores val into *addr.
// Consider using the more ergonomic and less error-prone [Pointer.Store] instead.
func StorePointer(addr *unsafe.Pointer, val unsafe.Pointer)
