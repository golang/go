// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.synctest

// Package synctest provides support for testing concurrent code.
//
// This package only exists when using Go compiled with GOEXPERIMENT=synctest.
// It is experimental, and not subject to the Go 1 compatibility promise.
package synctest

import (
	"internal/synctest"
)

// Run executes f in a new goroutine.
//
// The new goroutine and any goroutines transitively started by it form
// an isolated "bubble".
// Run waits for all goroutines in the bubble to exit before returning.
//
// Goroutines in the bubble use a synthetic time implementation.
// The initial time is midnight UTC 2000-01-01.
//
// Time advances when every goroutine in the bubble is blocked.
// For example, a call to time.Sleep will block until all other
// goroutines are blocked and return after the bubble's clock has
// advanced. See [Wait] for the specific definition of blocked.
//
// If every goroutine is blocked and there are no timers scheduled,
// Run panics.
//
// Channels, time.Timers, and time.Tickers created within the bubble
// are associated with it. Operating on a bubbled channel, timer, or ticker
// from outside the bubble panics.
func Run(f func()) {
	synctest.Run(f)
}

// Wait blocks until every goroutine within the current bubble,
// other than the current goroutine, is durably blocked.
// It panics if called from a non-bubbled goroutine,
// or if two goroutines in the same bubble call Wait at the same time.
//
// A goroutine is durably blocked if can only be unblocked by another
// goroutine in its bubble. The following operations durably block
// a goroutine:
//   - a send or receive on a channel from within the bubble
//   - a select statement where every case is a channel within the bubble
//   - sync.Cond.Wait
//   - time.Sleep
//
// A goroutine executing a system call or waiting for an external event
// such as a network operation is not durably blocked.
// For example, a goroutine blocked reading from an network connection
// is not durably blocked even if no data is currently available on the
// connection, because it may be unblocked by data written from outside
// the bubble or may be in the process of receiving data from a kernel
// network buffer.
//
// A goroutine is not durably blocked when blocked on a send or receive
// on a channel that was not created within its bubble, because it may
// be unblocked by a channel receive or send from outside its bubble.
func Wait() {
	synctest.Wait()
}
