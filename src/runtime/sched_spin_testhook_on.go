// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build runtime_test_sched

package runtime

import (
	"internal/runtime/atomic"
	_ "unsafe" // for go:linkname
)

const (
	spinExitIdle = iota
	spinExitArmed
	spinExitPaused
	spinExitWorkReady
	spinExitFoundWork
)

var spinExitTestState atomic.Uint32
var spinExitTestM atomic.Int64

// The last spinning M temporarily pauses with no P and no scheduler locks.
// The test's producer still owns its P and can enqueue work.
func schedTestSpinExit(mid int64) {
	if sched.nmspinning.Load() == 1 && spinExitTestState.CompareAndSwap(spinExitArmed, spinExitPaused) {
		spinExitTestM.Store(mid)
		for spinExitTestState.Load() == spinExitPaused {
			procyield(10)
		}
	}
}

// Track only the M whose transition the producer held open.
func schedTestSpinExitChecked(mid int64, found bool) {
	if found && spinExitTestM.Load() == mid {
		spinExitTestState.CompareAndSwap(spinExitWorkReady, spinExitFoundWork)
	}
}

var spinExitTestCompleted atomic.Uint32

//go:linkname schedTestRunSpinExitHandoff
func schedTestRunSpinExitHandoff() (paused, ran, observed bool) {
	spinExitTestCompleted.Store(0)
	spinExitTestM.Store(-1)
	mp := acquirem()
	defer releasem(mp)
	spinExitTestState.Store(spinExitArmed)
	defer spinExitTestState.Store(spinExitIdle)

	systemstack(wakep)
	const timeout = int64(2e9)
	for deadline := nanotime() + timeout; spinExitTestState.Load() != spinExitPaused; {
		if nanotime() >= deadline {
			return false, false, false
		}
		procyield(10)
	}

	// Enqueue on this P. wakep sees the existing spinner and leaves it
	// responsible for discovering the new runnable G. This goroutine keeps
	// its P until we have observed the other P execute our work.
	go func() { spinExitTestCompleted.Store(1) }()
	spinExitTestState.Store(spinExitWorkReady)
	for deadline := nanotime() + timeout; spinExitTestCompleted.Load() == 0; {
		if nanotime() >= deadline {
			return true, false, false
		}
		procyield(10)
	}
	return true, true, spinExitTestState.Load() == spinExitFoundWork
}
