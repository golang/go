// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "time"

// A Throttle permits throttling of a goroutine by
// calling the Throttle method repeatedly.
//
type Throttle struct {
	f  float64 // f = (1-r)/r for 0 < r < 1
	tm int64   // minimum run time slice; >= 0
	tr int64   // accumulated time running
	ts int64   // accumulated time stopped
	tt int64   // earliest throttle time (= time Throttle returned + tm)
}

// NewThrottle creates a new Throttle with a throttle value r and
// a minimum allocated run time slice of tm nanoseconds:
//
//	r == 0: "empty" throttle; the goroutine is always sleeping
//	r == 1: full throttle; the goroutine is never sleeping
//
// A value of r == 0.6 throttles a goroutine such that it runs
// approx. 60% of the time, and sleeps approx. 40% of the time.
// Values of r < 0 or r > 1 are clamped down to values between 0 and 1.
// Values of tm < 0 are set to 0.
//
func NewThrottle(r float64, tm int64) *Throttle {
	var f float64
	switch {
	case r <= 0:
		f = -1 // indicates always sleep
	case r >= 1:
		f = 0 // assume r == 1 (never sleep)
	default:
		// 0 < r < 1
		f = (1 - r) / r
	}
	if tm < 0 {
		tm = 0
	}
	return &Throttle{f: f, tm: tm, tt: time.Nanoseconds() + tm}
}

// Throttle calls time.Sleep such that over time the ratio tr/ts between
// accumulated run (tr) and sleep times (ts) approximates the value 1/(1-r)
// where r is the throttle value. Throttle returns immediately (w/o sleeping)
// if less than tm ns have passed since the last call to Throttle.
//
func (p *Throttle) Throttle() {
	if p.f < 0 {
		select {} // always sleep
	}

	t0 := time.Nanoseconds()
	if t0 < p.tt {
		return // keep running (minimum time slice not exhausted yet)
	}

	// accumulate running time
	p.tr += t0 - (p.tt - p.tm)

	// compute sleep time
	// Over time we want:
	//
	//	tr/ts = r/(1-r)
	//
	// Thus:
	//
	//	ts = tr*f with f = (1-r)/r
	//
	// After some incremental run time δr added to the total run time
	// tr, the incremental sleep-time δs to get to the same ratio again
	// after waking up from time.Sleep is:
	if δs := int64(float64(p.tr)*p.f) - p.ts; δs > 0 {
		time.Sleep(δs)
	}

	// accumulate (actual) sleep time
	t1 := time.Nanoseconds()
	p.ts += t1 - t0

	// set earliest next throttle time
	p.tt = t1 + p.tm
}
