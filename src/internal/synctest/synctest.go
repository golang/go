// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package synctest provides support for testing concurrent code.
//
// See the testing/synctest package for function documentation.
package synctest

import (
	"internal/abi"
	"unsafe"
)

//go:linkname Run
func Run(f func())

//go:linkname Wait
func Wait()

// IsInBubble reports whether the current goroutine is in a bubble.
//
//go:linkname IsInBubble
func IsInBubble() bool

// Association is the state of a pointer's bubble association.
type Association int

const (
	Unbubbled     = Association(iota) // not associated with any bubble
	CurrentBubble                     // associated with the current bubble
	OtherBubble                       // associated with a different bubble
)

// Associate attempts to associate p with the current bubble.
// It returns the new association status of p.
func Associate[T any](p *T) Association {
	// Ensure p escapes to permit us to attach a special to it.
	escapedP := abi.Escape(p)
	return Association(associate(unsafe.Pointer(escapedP)))
}

//go:linkname associate
func associate(p unsafe.Pointer) int

// Disassociate disassociates p from any bubble.
func Disassociate[T any](p *T) {
	disassociate(unsafe.Pointer(p))
}

//go:linkname disassociate
func disassociate(b unsafe.Pointer)

// IsAssociated reports whether p is associated with the current bubble.
func IsAssociated[T any](p *T) bool {
	return isAssociated(unsafe.Pointer(p))
}

//go:linkname isAssociated
func isAssociated(p unsafe.Pointer) bool

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
