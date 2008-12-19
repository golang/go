// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

import (
	"syscall";
	"time";
	"unsafe";
)

// TODO(rsc): This implementation of time.Tick is a
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
//			nsec, err := time.Nanoseconds();
//			c <- nsec;
//		}

func Ticker(ns int64, c chan int64) {
	var tv syscall.Timeval;
	now := time.Nanoseconds();
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

		syscall.nstotimeval(when - now, &tv);
		syscall.Syscall6(syscall.SYS_SELECT, 0, 0, 0, 0, int64(uintptr(unsafe.pointer(&tv))), 0);
		now = time.Nanoseconds();
		c <- now;
	}
}

export func Tick(ns int64) chan int64 {
	if ns <= 0 {
		return nil
	}
	c := new(chan int64);
	go Ticker(ns, c);
	return c;
}

