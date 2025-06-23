// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package synctest provides support for testing concurrent code.
//
// See the testing/synctest package for function documentation.
package synctest

import (
	_ "unsafe" // for go:linkname
)

//go:linkname Run
func Run(f func())

//go:linkname Wait
func Wait()

//go:linkname acquire
func acquire() any

//go:linkname release
func release(any)

//go:linkname inBubble
func inBubble(any, func())

// A Bubble is a synctest bubble.
//
// Not a public API. Used by syscall/js to propagate bubble membership through syscalls.
type Bubble struct {
	b any
}

// Acquire returns a reference to the current goroutine's bubble.
// The bubble will not become idle until Release is called.
func Acquire() *Bubble {
	if b := acquire(); b != nil {
		return &Bubble{b}
	}
	return nil
}

// Release releases the reference to the bubble,
// allowing it to become idle again.
func (b *Bubble) Release() {
	if b == nil {
		return
	}
	release(b.b)
	b.b = nil
}

// Run executes f in the bubble.
// The current goroutine must not be part of a bubble.
func (b *Bubble) Run(f func()) {
	if b == nil {
		f()
	} else {
		inBubble(b.b, f)
	}
}
