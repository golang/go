// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sync is the loGo/s shared synchronization support package:
// Locked[T], a mutex with panic-safe scoped access, and Hot[T], a
// lock-free read-mostly pointer with CAS updates and no retire list.
// .lgo code imports it as logos/sync; eject copies it into the shared
// support module github.com/gosuda/loGos/rt/sync (T-0.3).
//
// The loGo/s view differs in two signatures: Hot.Load returns
// const *T there, and Hot.Update takes func(old const *T) *T (SPEC
// T-206), each carried by a checker-side hook because Go packages
// emit no const facts (T-Sec6). Every other signature is identical
// in Go and loGo/s.
package sync

import (
	stdsync "sync"
	"sync/atomic"
)

// Locked pairs a value with its mutex. The zero value is ready to use.
// Access the value only through WithLock; copying a Locked after first
// use races exactly like copying a sync.Mutex.
type Locked[T any] struct {
	mu stdsync.Mutex
	v  T
}

// WithLock holds Lock, defers Unlock, then calls f with the value's
// address. The deferred Unlock runs during unwinding, so a panic in f
// unlocks the mutex before propagating (#176). f must neither return
// nor store its argument; .lgo callers get that rule from the M9 lint,
// which rejects escaping the pointer.
func (l *Locked[T]) WithLock(f func(*T)) {
	l.mu.Lock()
	defer l.mu.Unlock()
	f(&l.v)
}

// Hot is a read-mostly pointer: loads are one atomic load, updates
// swap in a fully constructed value with a CAS loop, so concurrent
// readers never observe a partially initialized value (#26). The zero
// value holds nil and is ready to use.
//
// There is no retire list and no background task: superseded values
// stay GC-managed, and stores run Go's hybrid write barrier through
// atomic.Pointer, so one million updates per second grow no memory
// (#153). Hot owns no goroutine and takes no cancellation signal;
// Update terminates once its CAS lands.
type Hot[T any] struct {
	p atomic.Pointer[T]
}

// Load returns the current value, or nil before the first Store. One
// atomic load; safe for concurrent use with Store and Update.
func (h *Hot[T]) Load() *T {
	return h.p.Load()
}

// Store publishes v atomically; readers after the store observe v.
// The value must stay unmodified once stored, or readers race with
// the mutator.
func (h *Hot[T]) Store(v *T) {
	h.p.Store(v)
}

// Update replaces the current value with f(old), retrying while a
// concurrent update lands first. f may run more than once, so it must
// be side-effect-free and must build its result from old without
// mutating it; only the returned value is published.
func (h *Hot[T]) Update(f func(old *T) *T) {
	for {
		old := h.p.Load()
		if h.p.CompareAndSwap(old, f(old)) {
			return
		}
	}
}
