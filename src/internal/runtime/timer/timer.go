// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package timer

// Timer is use to share the same timer value with runtime and time
//
// Code outside runtime has to be careful in using a timer value.
//
// Code that creates a new timer value can set the When, Period, F,
// Arg, and Seq fields.
// A new timer value may be passed to addtimer (called by time.startTimer).
// After doing that no fields may be touched.
//
// An active timer (one that has been passed to addtimer) may be
// passed to deltimer (time.stopTimer), after which it is no longer an
// active timer.  It is an inactive timer.
// In an inactive timer the period, f, arg, and seq fields may be modified,
// but not the when field.
// It's OK to just drop an inactive timer and let the GC collect it.
// It's not OK to pass an inactive timer to addtimer.
// Only newly allocated timer values may be passed to addtimer.
//
// An active timer may be passed to modtimer.  No fields may be touched.
// It remains an active timer.
//
// An inactive timer may be passed to resettimer to turn into an
// active timer with an updated when field.
// It's OK to pass a newly allocated timer value to resettimer.
//
// Timer operations are addtimer, deltimer, modtimer, resettimer,
// cleantimers, adjusttimers, and runtimer.
type Timer struct {
	// If this timer is on a heap, which P's heap it is on.
	// uintptr rather than *runtime.p because p isn't accessible from here.
	//
	// Only use at runtime.
	Pp uintptr

	// Timer wakes up at when, and then at when+period, ... (period > 0 only)
	// each time calling f(arg, now) in the timer goroutine, so f must be
	// a well-behaved function and not block.
	//
	// when must be positive on an active timer.
	When   int64
	Period int64
	F      func(any, uintptr)
	Arg    any
	Seq    uintptr

	// What to set the when field to in timerModifiedXX status.
	// Only use at runtime.
	Nextwhen int64

	// Status holds a value from runtime/time.go
	// Only use at runtime.
	// TODO: use atomic.Uint32
	Status uint32
}
