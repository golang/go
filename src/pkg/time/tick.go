// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

import (
	"syscall";
	"unsafe";
)

// TODO(rsc): This implementation of Tick is a
// simple placeholder.  Eventually, there will need to be
// a single central time server no matter how many tickers
// are active.  There also needs to be a way to cancel a ticker.
//
// Also, if timeouts become part of the select statement,
// perhaps the Ticker is just:
//
//	func Ticker(ns int64, c chan int64) {
//		for {
//			select { timeout ns: }
//			nsec, err := Nanoseconds();
//			c <- nsec;
//		}

func ticker(ns int64, c chan int64) {
	now := Nanoseconds();
	when := now;
	for {
		when += ns;	// next alarm

		// if c <- now took too long, skip ahead
		if when < now {
			// one big step
			when += (now-when)/ns * ns;
		}
		for when <= now {
			// little steps until when > now
			when += ns
		}

		Sleep(when - now);
		now = Nanoseconds();
		c <- now;
		if closed(c) {
			return;
		}
	}
}

// Tick creates a synchronous channel that will send the time, in nanoseconds,
// every ns nanoseconds.  It adjusts the intervals to make up for pauses in
// delivery of the ticks.
func Tick(ns int64) chan int64 {
	if ns <= 0 {
		return nil
	}
	c := make(chan int64);
	go ticker(ns, c);
	return c;
}

