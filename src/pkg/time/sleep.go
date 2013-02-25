// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

// Sleep pauses the current goroutine for the duration d.
func Sleep(d Duration)

func nano() int64 {
	sec, nsec := now()
	return sec*1e9 + int64(nsec)
}

// Interface to timers implemented in package runtime.
// Must be in sync with ../runtime/runtime.h:/^struct.Timer$
type runtimeTimer struct {
	i      int32
	when   int64
	period int64
	f      func(int64, interface{}) // NOTE: must not be closure
	arg    interface{}
}

// when is a helper function for setting the 'when' field of a runtimeTimer.
// It returns what the time will be, in nanoseconds, Duration d in the future.
// If d is negative, it is ignored.  If the returned value would be less than
// zero because of an overflow, MaxInt64 is returned.
func when(d Duration) int64 {
	if d <= 0 {
		return nano()
	}
	t := nano() + int64(d)
	if t < 0 {
		t = 1<<63 - 1 // math.MaxInt64
	}
	return t
}

func startTimer(*runtimeTimer)
func stopTimer(*runtimeTimer) bool

// The Timer type represents a single event.
// When the Timer expires, the current time will be sent on C,
// unless the Timer was created by AfterFunc.
type Timer struct {
	C <-chan Time
	r runtimeTimer
}

// Stop prevents the Timer from firing.
// It returns true if the call stops the timer, false if the timer has already
// expired or been stopped.
// Stop does not close the channel, to prevent a read from the channel succeeding
// incorrectly.
func (t *Timer) Stop() bool {
	return stopTimer(&t.r)
}

// NewTimer creates a new Timer that will send
// the current time on its channel after at least duration d.
func NewTimer(d Duration) *Timer {
	c := make(chan Time, 1)
	t := &Timer{
		C: c,
		r: runtimeTimer{
			when: when(d),
			f:    sendTime,
			arg:  c,
		},
	}
	startTimer(&t.r)
	return t
}

// Reset changes the timer to expire after duration d.
// It returns true if the timer had been active, false if the timer had
// expired or been stopped.
func (t *Timer) Reset(d Duration) bool {
	w := when(d)
	active := stopTimer(&t.r)
	t.r.when = w
	startTimer(&t.r)
	return active
}

func sendTime(now int64, c interface{}) {
	// Non-blocking send of time on c.
	// Used in NewTimer, it cannot block anyway (buffer).
	// Used in NewTicker, dropping sends on the floor is
	// the desired behavior when the reader gets behind,
	// because the sends are periodic.
	select {
	case c.(chan Time) <- Unix(0, now):
	default:
	}
}

// After waits for the duration to elapse and then sends the current time
// on the returned channel.
// It is equivalent to NewTimer(d).C.
func After(d Duration) <-chan Time {
	return NewTimer(d).C
}

// AfterFunc waits for the duration to elapse and then calls f
// in its own goroutine. It returns a Timer that can
// be used to cancel the call using its Stop method.
func AfterFunc(d Duration, f func()) *Timer {
	t := &Timer{
		r: runtimeTimer{
			when: when(d),
			f:    goFunc,
			arg:  f,
		},
	}
	startTimer(&t.r)
	return t
}

func goFunc(now int64, arg interface{}) {
	go arg.(func())()
}
