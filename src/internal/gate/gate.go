// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gate contains an alternative condition variable.
package gate

import "context"

// A Gate is a monitor (mutex + condition variable) with one bit of state.
//
// The condition may be either set or unset.
// Lock operations may be unconditional, or wait for the condition to be set.
// Unlock operations record the new state of the condition.
type Gate struct {
	// When unlocked, exactly one of set or unset contains a value.
	// When locked, neither chan contains a value.
	set   chan struct{}
	unset chan struct{}
}

// New returns a new, unlocked gate.
func New(set bool) Gate {
	g := Gate{
		set:   make(chan struct{}, 1),
		unset: make(chan struct{}, 1),
	}
	g.Unlock(set)
	return g
}

// Lock acquires the gate unconditionally.
// It reports whether the condition is set.
func (g *Gate) Lock() (set bool) {
	select {
	case <-g.set:
		return true
	case <-g.unset:
		return false
	}
}

// WaitAndLock waits until the condition is set before acquiring the gate.
// If the context expires, WaitAndLock returns an error and does not acquire the gate.
func (g *Gate) WaitAndLock(ctx context.Context) error {
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

// LockIfSet acquires the gate if and only if the condition is set.
func (g *Gate) LockIfSet() (acquired bool) {
	select {
	case <-g.set:
		return true
	default:
		return false
	}
}

// Unlock sets the condition and releases the gate.
func (g *Gate) Unlock(set bool) {
	if set {
		g.set <- struct{}{}
	} else {
		g.unset <- struct{}{}
	}
}
