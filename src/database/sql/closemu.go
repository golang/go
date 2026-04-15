// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sql

import (
	"sync"
	"sync/atomic"
)

// A closingMutex is an RWMutex for synchronizing close.
// Unlike a sync.RWMutex, RLock takes priority over Lock.
// Reads can starve out close, but reads are safely reentrant.
type closingMutex struct {
	// state is 2*readers+writerWaiting.
	//   0 is unlocked
	//   1 is unlocked and a writer needs to wake
	//   >0 is read-locked
	//   <0 is write-locked
	state atomic.Int64
	mu    sync.Mutex
	read  *sync.Cond
	write *sync.Cond
}

func (m *closingMutex) RLock() {
	if m.TryRLock() {
		return
	}

	// Wait for writer.
	m.mu.Lock()
	defer m.mu.Unlock()
	for {
		if m.TryRLock() {
			return
		}
		m.init()
		m.read.Wait()
	}
}

func (m *closingMutex) RUnlock() {
	for {
		x := m.state.Load()
		if x < 2 {
			panic("runlock of un-rlocked mutex")
		}
		if m.state.CompareAndSwap(x, x-2) {
			if x-2 == 1 {
				// We were the last reader, and a writer is waiting.
				// The lock makes sure the writer sees the broadcast.
				m.mu.Lock()
				defer m.mu.Unlock()
				m.write.Broadcast()
			}
			return
		}
	}
}

func (m *closingMutex) Lock() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for {
		x := m.state.Load()
		if (x == 0 || x == 1) && m.state.CompareAndSwap(x, -1) {
			return
		}
		// Set writer waiting bit and sleep.
		if x&1 == 0 && !m.state.CompareAndSwap(x, x|1) {
			continue
		}
		m.init()
		m.write.Wait()
	}
}

func (m *closingMutex) Unlock() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.state.CompareAndSwap(-1, 0) {
		panic("unlock of unlocked mutex")
	}
	if m.read != nil {
		m.read.Broadcast()
		m.write.Broadcast()
	}
}

func (m *closingMutex) TryRLock() bool {
	for {
		x := m.state.Load()
		if x < 0 {
			return false
		}
		if m.state.CompareAndSwap(x, x+2) {
			return true
		}
	}
}

func (m *closingMutex) init() {
	// Lazily create the read/write Conds.
	// In the common, uncontended case, we'll never need them.
	if m.read == nil {
		m.read = sync.NewCond(&m.mu)
		m.write = sync.NewCond(&m.mu)
	}
}
