// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync

import (
	"sync/atomic"
	"unsafe"
)

// Counter is a concurrent counter of int64, uint64, or float64 values.
//
// Counter is specialized for frequent, contended writes and infrequent
// reads. Prefer Counter over a [sync/atomic] or [Mutex] protected counter for
// improved performance when writes are much more common than reads,
// and approximate (unsynchronized) reads are acceptable for the
// use case. Value takes time proportional to the maximum GOMAXPROCS
// seen by the counter.
//
// The zero Counter has value 0 and is ready for use.
// A Counter must not be copied after first use.
type Counter[T ~float64 | ~int64 | ~uint64] struct {
	cache stripe
}

// Add adds delta to the counter.
//
// In the terminology of [the Go memory model], Add is NOT synchronized before
// calls to Add or [Counter.Value] on other goroutines.
// Callers must not rely on Add as a memory barrier or synchronization
// primitive to publish or synchronize access to other non-counter memory
// mutations.
//
// For integer types, overflow wraps around, the same as [integer overflow]
// defined in the Go Language Specification.
//
// For floating-point types, additions are accumulated in arbitrary
// order, so rounding and overflow may differ from sequential addition.
//
// [the Go memory model]: https://go.dev/ref/mem
// [integer overflow]: https://go.dev/ref/spec#Integer_overflow
func (c *Counter[T]) Add(delta T) {
	// Pinning to a P ensures no other goroutine writes to the
	// returned shard until we unpin, so the load need not be atomic.
	pID := runtime_procPin()
	n := c.cache.get(pID)
	newVal := *(*T)(unsafe.Pointer(n)) + delta
	// Store atomically so concurrent Value loads cannot tear or race.
	atomic.StoreUint64(n, *(*uint64)(unsafe.Pointer(&newVal)))
	runtime_procUnpin()
}

// Value returns the sum of the values added to the counter.
//
// Value and [Counter.Add] may be called concurrently.
//
// In the terminology of [the Go memory model], [Counter.Add] is NOT
// synchronized before Value. In other words, Value observes an arbitrary
// subset of concurrent [Counter.Add] calls. Informally, Value returns only an
// approximate sum of all [Counter.Add] calls.
//
// While callers may use external synchronization to make [Counter.Add]
// synchronized before Value, note that the sum still occurs in an
// arbitrary order. For float64, the value may still differ from
// sequential addition due to rounding or overflow.
//
// Value takes time proportional to GOMAXPROCS.
//
// [the Go memory model]: https://go.dev/ref/mem
func (c *Counter[T]) Value() T {
	var sum T
	s := &c.cache
	stripeCap := s.cap()
	block := unsafe.Slice(s.block.Load(), stripeCap)

	for i := range stripeCap {
		n := block[i]
		// Load atomically to prevent tearing and races with concurrent Add.
		v := atomic.LoadUint64(n)
		sum += *(*T)(unsafe.Pointer(&v))
	}
	return sum
}
