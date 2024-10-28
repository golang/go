// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mainthread mediates access to the program's main thread.
//
// Most Go programs do not need to run on specific threads
// and can ignore this package, but some C libraries, often GUI-related libraries,
// only work when invoked from the program's main thread.
//
// [Do] runs a function on the main thread. No other code can run on the main thread
// until that function returns. If the function is long-running, it can
// yield the main thread temporarily to other [Do] calls by calling [Yield].
//
// Each package's initialization functions always run on the main thread,
// so call Do in init is unnecessary.
//
// For compatibility with earlier versions of Go, if an init function calls [runtime.LockOSThread],
// then package main's func main also runs on the main thread,
// until the thread is unlocked using [runtime.UnlockOSThread].
// In this situation, main must explicitly yield the main thread
// to allow other thread calls to Do are to proceed.
// See the documentation for [Waiting] for examples.
package mainthread

import _ "unsafe"

// Do calls f on the main thread.
// Nothing else runs on the main thread until f returns or calls [Yield].
// If f calls Do, the nested call panics.
//
//go:linkname Do runtime.mainThreadDo
func Do(f func())

// Yield yields the main thread in turn to every other [Do] (or returning Yield)
// call that is blocked waiting for the main thread when Yield is called.
// It then waits to reacquire the main thread and returns.
//
// Yield must only be called from the main thread (during a call to Do).
// If called from a different thread, Yield panics.
//
//go:linkname Yield runtime.mainThreadYield
func Yield()

// Waiting returns a channel that receives a message when a call to [Do]
// is blocked waiting for the main thread. Multiple blocked calls to Do may be
// coalesced into a single message.
// There is only one waiting channel; all calls to Waiting return the same channel.
//
// Programs that run a C-based API such as a GUI event loop on the main thread
// should arrange to share it by watching for events on Waiting and calling [Yield].
// A typical approach is to define a new event type that can be sent to the event loop,
// respond to that event in the loop by calling Yield,
// and then start a separate goroutine (running on a non-main thread)
// that watches the waiting channel and signals the event loop:
//
//	go func() {
//	    for range mainthread.Waiting() {
//	        sendYieldEvent()
//	    }
//	}()
//
//	C.EventLoop()
//
//go:linkname Waiting runtime.mainThreadWaiting
func Waiting() <-chan struct{}
