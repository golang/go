// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync

import _ "unsafe"

// defined in package runtime

// SemacquireMutex is like Semacquire, but for profiling contended
// Mutexes and RWMutexes.
// If lifo is true, queue waiter at the head of wait queue.
// skipframes is the number of frames to omit during tracing, counting from
// runtime_SemacquireMutex's caller.
// The different forms of this function just tell the runtime how to present
// the reason for waiting in a backtrace, and is used to compute some metrics.
// Otherwise they're functionally identical.
//
//go:linkname runtime_SemacquireMutex
func runtime_SemacquireMutex(s *uint32, lifo bool, skipframes int)

// Semrelease atomically increments *s and notifies a waiting goroutine
// if one is blocked in Semacquire.
// It is intended as a simple wakeup primitive for use by the synchronization
// library and should not be used directly.
// If handoff is true, pass count directly to the first waiter.
// skipframes is the number of frames to omit during tracing, counting from
// runtime_Semrelease's caller.
//
//go:linkname runtime_Semrelease
func runtime_Semrelease(s *uint32, handoff bool, skipframes int)

// Active spinning runtime support.
// runtime_canSpin reports whether spinning makes sense at the moment.
//
//go:linkname runtime_canSpin
func runtime_canSpin(i int) bool

// runtime_doSpin does active spinning.
//
//go:linkname runtime_doSpin
func runtime_doSpin()

//go:linkname runtime_nanotime
func runtime_nanotime() int64

//go:linkname throw
func throw(string)

//go:linkname fatal
func fatal(string)
