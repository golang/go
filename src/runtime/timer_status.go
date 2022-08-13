// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Time-related runtime and pieces of package time.

package runtime

// timer.start():
//   timerNoStatus   -> timerWaiting
//   anything else   -> panic: invalid value
// timer.stop():
//   timerWaiting         -> timerModifying -> timerDeleted
//   timerModifiedEarlier -> timerModifying -> timerDeleted
//   timerModifiedLater   -> timerModifying -> timerDeleted
//   timerNoStatus        -> do nothing
//   timerDeleted         -> do nothing
//   timerRemoving        -> do nothing
//   timerRemoved         -> do nothing
//   timerRunning         -> wait until status changes
//   timerMoving          -> wait until status changes
//   timerModifying       -> wait until status changes
// timer.modify():
//   timerWaiting    -> timerModifying -> timerModifiedXX
//   timerModifiedXX -> timerModifying -> timerModifiedYY
//   timerNoStatus   -> timerModifying -> timerWaiting
//   timerRemoved    -> timerModifying -> timerWaiting
//   timerDeleted    -> timerModifying -> timerModifiedXX
//   timerRunning    -> wait until status changes
//   timerMoving     -> wait until status changes
//   timerRemoving   -> wait until status changes
//   timerModifying  -> wait until status changes
// 
// timing.cleanTimers() (looks in P's timer heap):
//   timerDeleted    -> timerRemoving -> timerRemoved
//   timerModifiedXX -> timerMoving -> timerWaiting
// timing.adjustTimers() (looks in P's timer heap):
//   timerDeleted    -> timerRemoving -> timerRemoved
//   timerModifiedXX -> timerMoving -> timerWaiting
// timing.runTimer() (looks in P's timer heap):
//   timerNoStatus   -> panic: uninitialized timer
//   timerWaiting    -> timerWaiting or
//   timerWaiting    -> timerRunning -> timerNoStatus or
//   timerWaiting    -> timerRunning -> timerWaiting
//   timerModifying  -> wait until status changes
//   timerModifiedXX -> timerMoving -> timerWaiting
//   timerDeleted    -> timerRemoving -> timerRemoved
//   timerRunning    -> panic: concurrent runTimer calls
//   timerRemoved    -> panic: inconsistent timer heap
//   timerRemoving   -> panic: inconsistent timer heap
//   timerMoving     -> panic: inconsistent timer heap

// Values for the timer status field.
const (
	// Timer has no status set yet.
	timerNoStatus = iota

	// Waiting for timer to fire.
	// The timer is in some P's heap.
	timerWaiting

	// Running the timer function.
	// A timer will only have this status briefly.
	timerRunning

	// The timer is deleted and should be removed.
	// It should not be run, but it is still in some P's heap.
	timerDeleted

	// The timer is being removed.
	// The timer will only have this status briefly.
	timerRemoving

	// The timer has been stopped.
	// It is not in any P's heap.
	timerRemoved

	// The timer is being modified.
	// The timer will only have this status briefly.
	timerModifying

	// The timer has been modified to an earlier time.
	// The new when value is in the nextwhen field.
	// The timer is in some P's heap, possibly in the wrong place.
	timerModifiedEarlier

	// The timer has been modified to the same or a later time.
	// The new when value is in the nextwhen field.
	// The timer is in some P's heap, possibly in the wrong place.
	timerModifiedLater

	// The timer has been modified and is being moved.
	// The timer will only have this status briefly.
	timerMoving
)
