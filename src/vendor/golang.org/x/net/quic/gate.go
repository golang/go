// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import "context"

// An gate is a monitor (mutex + condition variable) with one bit of state.
//
// The condition may be either set or unset.
// Lock operations may be unconditional, or wait for the condition to be set.
// Unlock operations record the new state of the condition.
type gate struct {
	// When unlocked, exactly one of set or unset contains a value.
	// When locked, neither chan contains a value.
	set   chan struct{}
	unset chan struct{}
}

// newGate returns a new, unlocked gate with the condition unset.
func newGate() gate {
	g := newLockedGate()
	g.unlock(false)
	return g
}

// newLockedGate returns a new, locked gate.
func newLockedGate() gate {
	return gate{
		set:   make(chan struct{}, 1),
		unset: make(chan struct{}, 1),
	}
}

// lock acquires the gate unconditionally.
// It reports whether the condition is set.
func (g *gate) lock() (set bool) {
	select {
	case <-g.set:
		return true
	case <-g.unset:
		return false
	}
}

// waitAndLock waits until the condition is set before acquiring the gate.
// If the context expires, waitAndLock returns an error and does not acquire the gate.
func (g *gate) waitAndLock(ctx context.Context) error {
	select {
	case <-g.set:
		return nil
	default:
	}
	select {
	case <-g.set:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// lockIfSet acquires the gate if and only if the condition is set.
func (g *gate) lockIfSet() (acquired bool) {
	select {
	case <-g.set:
		return true
	default:
		return false
	}
}

// unlock sets the condition and releases the gate.
func (g *gate) unlock(set bool) {
	if set {
		g.set <- struct{}{}
	} else {
		g.unset <- struct{}{}
	}
}

// unlockFunc sets the condition to the result of f and releases the gate.
// Useful in defers.
func (g *gate) unlockFunc(f func() bool) {
	g.unlock(f())
}
