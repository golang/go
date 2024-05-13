// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package iter provides basic definitions and operations
// related to iteration in Go.
package iter

import (
	"internal/race"
	"unsafe"
)

// Seq is an iterator over sequences of individual values.
// When called as seq(yield), seq calls yield(v) for each value v in the sequence,
// stopping early if yield returns false.
type Seq[V any] func(yield func(V) bool)

// Seq2 is an iterator over sequences of pairs of values, most commonly key-value pairs.
// When called as seq(yield), seq calls yield(k, v) for each pair (k, v) in the sequence,
// stopping early if yield returns false.
type Seq2[K, V any] func(yield func(K, V) bool)

type coro struct{}

//go:linkname newcoro runtime.newcoro
func newcoro(func(*coro)) *coro

//go:linkname coroswitch runtime.coroswitch
func coroswitch(*coro)

// Pull converts the “push-style” iterator sequence seq
// into a “pull-style” iterator accessed by the two functions
// next and stop.
//
// Next returns the next value in the sequence
// and a boolean indicating whether the value is valid.
// When the sequence is over, next returns the zero V and false.
// It is valid to call next after reaching the end of the sequence
// or after calling stop. These calls will continue
// to return the zero V and false.
//
// Stop ends the iteration. It must be called when the caller is
// no longer interested in next values and next has not yet
// signaled that the sequence is over (with a false boolean return).
// It is valid to call stop multiple times and when next has
// already returned false.
//
// It is an error to call next or stop from multiple goroutines
// simultaneously.
func Pull[V any](seq Seq[V]) (next func() (V, bool), stop func()) {
	var (
		v         V
		ok        bool
		done      bool
		yieldNext bool
		racer     int
	)
	c := newcoro(func(c *coro) {
		race.Acquire(unsafe.Pointer(&racer))
		yield := func(v1 V) bool {
			if done {
				return false
			}
			if !yieldNext {
				panic("iter.Pull: yield called again before next")
			}
			yieldNext = false
			v, ok = v1, true
			race.Release(unsafe.Pointer(&racer))
			coroswitch(c)
			race.Acquire(unsafe.Pointer(&racer))
			return !done
		}
		seq(yield)
		var v0 V
		v, ok = v0, false
		done = true
		race.Release(unsafe.Pointer(&racer))
	})
	next = func() (v1 V, ok1 bool) {
		race.Write(unsafe.Pointer(&racer)) // detect races
		if done {
			return
		}
		if yieldNext {
			panic("iter.Pull: next called again before yield")
		}
		yieldNext = true
		race.Release(unsafe.Pointer(&racer))
		coroswitch(c)
		race.Acquire(unsafe.Pointer(&racer))
		return v, ok
	}
	stop = func() {
		race.Write(unsafe.Pointer(&racer)) // detect races
		if !done {
			done = true
			race.Release(unsafe.Pointer(&racer))
			coroswitch(c)
			race.Acquire(unsafe.Pointer(&racer))
		}
	}
	return next, stop
}

// Pull2 converts the “push-style” iterator sequence seq
// into a “pull-style” iterator accessed by the two functions
// next and stop.
//
// Next returns the next pair in the sequence
// and a boolean indicating whether the pair is valid.
// When the sequence is over, next returns a pair of zero values and false.
// It is valid to call next after reaching the end of the sequence
// or after calling stop. These calls will continue
// to return a pair of zero values and false.
//
// Stop ends the iteration. It must be called when the caller is
// no longer interested in next values and next has not yet
// signaled that the sequence is over (with a false boolean return).
// It is valid to call stop multiple times and when next has
// already returned false.
//
// It is an error to call next or stop from multiple goroutines
// simultaneously.
func Pull2[K, V any](seq Seq2[K, V]) (next func() (K, V, bool), stop func()) {
	var (
		k         K
		v         V
		ok        bool
		done      bool
		yieldNext bool
		racer     int
	)
	c := newcoro(func(c *coro) {
		race.Acquire(unsafe.Pointer(&racer))
		yield := func(k1 K, v1 V) bool {
			if done {
				return false
			}
			if !yieldNext {
				panic("iter.Pull2: yield called again before next")
			}
			yieldNext = false
			k, v, ok = k1, v1, true
			race.Release(unsafe.Pointer(&racer))
			coroswitch(c)
			race.Acquire(unsafe.Pointer(&racer))
			return !done
		}
		seq(yield)
		var k0 K
		var v0 V
		k, v, ok = k0, v0, false
		done = true
		race.Release(unsafe.Pointer(&racer))
	})
	next = func() (k1 K, v1 V, ok1 bool) {
		race.Write(unsafe.Pointer(&racer)) // detect races
		if done {
			return
		}
		if yieldNext {
			panic("iter.Pull2: next called again before yield")
		}
		yieldNext = true
		race.Release(unsafe.Pointer(&racer))
		coroswitch(c)
		race.Acquire(unsafe.Pointer(&racer))
		return k, v, ok
	}
	stop = func() {
		race.Write(unsafe.Pointer(&racer)) // detect races
		if !done {
			done = true
			race.Release(unsafe.Pointer(&racer))
			coroswitch(c)
			race.Acquire(unsafe.Pointer(&racer))
		}
	}
	return next, stop
}
